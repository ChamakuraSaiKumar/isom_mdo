"""
=============================================================================
IDF ANALYSIS DASHBOARD – CLOUD DEPLOYMENT VERSION
(Zero-Compute Online Mode. Sanitizes SOM/OpenMDAO objects into pure DataFrames)
=============================================================================
"""
import os
import pickle
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  
import plotly.graph_objects as go
import plotly.express as px
import html
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.layouts import column as bk_col, row as bk_row
from bokeh.models import Div

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
COL = {
    'Valid Space (D + F)': '#2ca02c', 'Decoupled Only': '#1f77b4',
    'Feasible Only': '#ff7f0e', 'Neither': '#d62728', 'iSOM Anchor': '#9467bd'
}
OBJ_TOL = 1e-3
TOL_CONSTRAINT = 1e-6

# =============================================================================
# 2. HTML RENDERERS (Only used during local generation)
# =============================================================================
def _render_tracker_html_fixed(snap, an_obj):
    from sompy.visualization.viz_functions import Visualization_func
    from isom_mdo_scalable import _resize
    obj_grid, tau_vals = snap["Obj"], snap["tau"]
    vis = Visualization_func(an_obj.sm)
    tau_text = " | ".join([f"{k}:{v:.4e}" for k, v in tau_vals.items()])
    d_min, d_max = np.nanmin(obj_grid), np.nanmax(obj_grid)
    norm_grid = ((obj_grid - d_min) / (d_max - d_min + 1e-12) if d_max > d_min else obj_grid)
    norm_grid = np.nan_to_num(norm_grid, nan=0.0)
 
    def make_panel(mask_grid, title, color):
        orig_col = an_obj.sm.codebook.matrix[:, 0].copy()
        an_obj.sm.codebook.matrix[:, 0] = norm_grid.flatten()
        hits = mask_grid.astype(int).flatten()
        p = vis.plot_hitmap(hits, comp=0, clr=color)
        an_obj.sm.codebook.matrix[:, 0] = orig_col 
        p.title.text = f"{title}"
        p.title.align, p.title.text_font_size = "center", "10pt"
        return _resize(p, w=300, h=300)
 
    p_dec  = make_panel(snap["D_Intersect"], f"Decoupled ({int(snap['D_Intersect'].sum())})", "purple")
    p_feas = make_panel(snap["F_Intersect"], f"Feasible ({int(snap['F_Intersect'].sum())})",  "darkgreen")
    p_valid= make_panel(snap["Valid"], f"Valid ({snap['n_valid']})", "gray")
    p_opt  = make_panel(snap["Converged"], f"Optima ({snap['n_converged']})", "red")
 
    header = Div(text=(f"<h2 style='color:#333;text-align:center;margin:8px 0'>Iteration: {snap['iteration']} &nbsp;&nbsp;|&nbsp;&nbsp; τ → {tau_text}</h2>"), width=1220)
    layout = bk_col(header, bk_row(p_dec, p_feas, p_valid, p_opt))
    return file_html(layout, CDN, title=f"Iter {snap['iteration']}")

def build_zero_flicker_player(html_frames):
    iframes_html = ""
    for i, frame_html in enumerate(html_frames):
        safe_html = html.escape(frame_html)
        display = "opacity: 1; z-index: 2;" if i == 0 else "opacity: 0; z-index: 1;"
        iframes_html += f"<iframe id='frame_{i}' srcdoc='{safe_html}' style='position:absolute; top:0; left:0; width:100%; height:100%; border:none; {display}'></iframe>\n"

    return f"""
    <html><head><style>
        body {{ font-family: sans-serif; margin: 0; overflow: hidden; }}
        .controls {{ display: flex; align-items: center; justify-content: center; gap: 15px; padding: 10px; background: #f8f9fa; border-bottom: 1px solid #ddd; }}
        button {{ padding: 8px 15px; cursor: pointer; border: none; background: #007bff; color: white; border-radius: 4px; font-weight: bold; font-size: 14px;}}
        button:hover {{ background: #0056b3; }}
        .screen-container {{ position: relative; width: 100%; height: 500px; }}
    </style></head><body>
        <div class="controls">
            <button onclick="playVideo()">▶ Play</button>
            <button onclick="pauseVideo()">⏸ Pause</button>
            <input type="range" id="frame-slider" min="0" max="{len(html_frames)-1}" value="0" oninput="goToFrame(this.value)" style="width: 300px;">
            <span id="frame-counter" style="font-weight: bold;">Iter: 0 / {len(html_frames)-1}</span>
        </div>
        <div class="screen-container">{iframes_html}</div>
        <script>
            let currentFrame = 0; const totalFrames = {len(html_frames)}; let isPlaying = false; let timer = null;
            function showFrame(index) {{
                for(let i=0; i<totalFrames; i++) {{
                    let f = document.getElementById('frame_' + i);
                    if(i === parseInt(index)) {{ f.style.opacity = '1'; f.style.zIndex = '2'; }} 
                    else {{ f.style.opacity = '0'; f.style.zIndex = '1'; }}
                }}
                document.getElementById('frame-slider').value = index;
                document.getElementById('frame-counter').innerText = "Iter: " + index + " / " + (totalFrames-1);
            }}
            function nextFrame() {{ currentFrame++; if(currentFrame >= totalFrames) {{ pauseVideo(); currentFrame = totalFrames - 1; }} showFrame(currentFrame); }}
            function playVideo() {{ if(isPlaying) return; if(currentFrame >= totalFrames - 1) currentFrame = -1; isPlaying = true; timer = setInterval(nextFrame, 400); }}
            function pauseVideo() {{ isPlaying = false; clearInterval(timer); }}
            function goToFrame(index) {{ pauseVideo(); currentFrame = parseInt(index); showFrame(currentFrame); }}
        </script>
    </body></html>
    """

def check_boolean_regions(c_vals, c_cols, tau_dict, g_vals, constraints_logic):
    is_dec = all(abs(c_vals[i]) < tau_dict[c_cols[i]] for i in range(len(c_cols)))
    is_feas = True
    for g_key, (op, val) in constraints_logic.items():
        g = g_vals[g_key]
        if op in (">=", ">") and not (g >= val - TOL_CONSTRAINT): is_feas = False
        if op in ("<=", "<") and not (g <= val + TOL_CONSTRAINT): is_feas = False
    return is_dec, is_feas

# =============================================================================
# 3. CORE LOGIC (RUNS LOCALLY ONLY)
# =============================================================================
@st.cache_resource(show_spinner=False)
def compute_everything_upfront(_prob_mod, n_samples, som_grid_size):
    """Loads cache online. Generates and sanitizes data locally."""
    os.makedirs("results", exist_ok=True)
    cache_file = f"results/{_prob_mod.__name__}_CloudCache_{som_grid_size[0]}x{som_grid_size[1]}.pkl"
    
    # ---------------------------------------------------------
    # CLOUD MODE: LOAD FROM PICKLE (INSTANT)
    # ---------------------------------------------------------
    if os.path.exists(cache_file):
        with st.spinner("☁️ Loading Precomputed Data from Cache..."):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
                
    # ---------------------------------------------------------
    # LOCAL MODE: GENERATE DATA
    # ---------------------------------------------------------
    from isom_mdo_scalable import ISOMAnalyzer, run_dashboard
    results_dict = {1: {}, 2: {}, 3: {}}
    m_rows, m_cols = som_grid_size
    total_nodes = m_rows * m_cols
    c_vars, g_vars, d_vars = _prob_mod.METADATA["consistency_vars"], _prob_mod.METADATA["constraint_vars"], _prob_mod.METADATA["design_vars"]
    BOUNDS = getattr(_prob_mod, "SOM_BOUNDS", {})
    TAU_FLOOR_VALS = getattr(_prob_mod, 'TAU_FLOOR', {c: 1e-7 for c in c_vars})
    n_jobs = min(8, multiprocessing.cpu_count())

    with st.status(f"🖥️ Local Generation: Pre-computing ALL 9 Variations ({total_nodes} nodes each)...", expanded=True) as status:
        df_full = _prob_mod.generate_full_dataset(N=n_samples)

        for mode in [1, 2, 3]:
            st.write(f"▶ Processing Case {mode}...")
            results_dict[mode] = {'methods': {}}
            an = ISOMAnalyzer(df_full, _prob_mod.METADATA, mode, som_grid_size)
            an.train(train_rough_len=5, train_finetune_len=5)
            an.evaluate(_prob_mod.evaluate_sellar_physics)
            an.compute_regions(20, _prob_mod.CONSTRAINTS_LOGIC)

            # Read the Bokeh HTML file content directly into memory
            html_path = run_dashboard(an, problem_name=f"{_prob_mod.__name__}_Case_{mode}", show_results=False)
            with open(html_path, "r", encoding="utf-8") as f:
                results_dict[mode]['bokeh_html_string'] = f.read()

            def run_node(i, j):
                start = {}
                for var in d_vars:
                    val = float(an.W_grid[i, j, an.trained_cols.index(var)])
                    if var in BOUNDS: val = float(np.clip(val, BOUNDS[var][0], BOUNDS[var][1]))
                    start[var] = val
                return i, j, start, _prob_mod.run_optimizer(start)

            all_trajectories = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
                futures = {ex.submit(run_node, i, j): (i, j) for i in range(m_rows) for j in range(m_cols)}
                for fut in concurrent.futures.as_completed(futures):
                    i, j = futures[fut]
                    _, _, start_dict, res = fut.result()
                    history = res.get("history", [])
                    if not history: history = [start_dict]
                    all_trajectories.append({
                        "node_i": i, "node_j": j, "node_id": i * m_cols + j,
                        "history": history, "converged": res.get("converged", False),
                        "converged_at": len(history) - 1 if res.get("converged") else -1,
                        "n_iterations": len(history), "start_norm": an.Coupling_error[i, j],
                        "d_dict": start_dict, "opt_res": res
                    })

            max_iters = max(len(t["history"]) for t in all_trajectories)
            grids_c = {it: {c: np.full((m_rows, m_cols), np.nan) for c in c_vars} for it in range(max_iters)}
            grids_g = {it: {g: np.full((m_rows, m_cols), np.nan) for g in g_vars} for it in range(max_iters)}
            grids_obj = {it: np.full((m_rows, m_cols), np.nan) for it in range(max_iters)}
            conv_masks = {it: np.zeros((m_rows, m_cols), dtype=bool) for it in range(max_iters)}

            for traj in all_trajectories:
                i, j = traj["node_i"], traj["node_j"]
                for it in range(max_iters):
                    state = (traj["history"][it] if it < len(traj["history"]) else traj["history"][-1]).copy()
                    if it == 0 and mode == 1:
                        for c in c_vars: grids_c[it][c][i, j] = an.C_grids[c][i, j]
                        for g in g_vars: grids_g[it][g][i, j] = an.g_grids[g][i, j]
                        grids_obj[it][i, j] = an.Obj_grid[i, j]
                    else:
                        d_state = {var: state[var] for var in d_vars if var in state}
                        phys = _prob_mod.evaluate_sellar_physics(d_state)
                        for c in c_vars: grids_c[it][c][i, j] = phys.get(c, 0.0)
                        for g in g_vars: grids_g[it][g][i, j] = phys.get(g, 0.0)
                        grids_obj[it][i, j] = state.get("obj", phys.get(_prob_mod.METADATA["objective_var"], np.nan))
                    if traj["converged"] and 0 <= traj["converged_at"] <= it:
                        conv_masks[it][i, j] = True

            TAU_VAR, TAU_FLOOR = [], []
            for it in range(max_iters):
                t_v, t_f = {}, {}
                for c in c_vars:
                    vals = np.abs(grids_c[it][c].flatten()); vals = vals[~np.isnan(vals)]
                    t_v[c] = float(np.percentile(vals, 20))
                    t_f[c] = max(t_v[c], TAU_FLOOR_VALS.get(c, 1e-7))
                TAU_VAR.append(t_v); TAU_FLOOR.append(t_f)
            
            methods_tau_map = {'Fixed': [TAU_VAR[0]]*max_iters, 'Variable': TAU_VAR, 'Floor': TAU_FLOOR}

            for method_name, tau_hist in methods_tau_map.items():
                snapshots = []
                for it in range(max_iters):
                    D_Int = np.logical_and.reduce([np.abs(grids_c[it][c]) < tau_hist[it][c] for c in c_vars])
                    F_msks = [grids_g[it][k] >= (v - TOL_CONSTRAINT) if op in (">=", ">") else grids_g[it][k] <= (v + TOL_CONSTRAINT) for k, (op, v) in _prob_mod.CONSTRAINTS_LOGIC.items()]
                    F_Int = np.logical_and.reduce(F_msks) if F_msks else np.ones((m_rows, m_cols), dtype=bool)
                    snapshots.append({
                        "iteration": it, "tau": tau_hist[it], "Obj": grids_obj[it],
                        "D_Intersect": D_Int, "F_Intersect": F_Int, "Valid": D_Int & F_Int, 
                        "Converged": conv_masks[it], "n_valid": int(np.sum(D_Int & F_Int)), "n_converged": int(np.sum(conv_masks[it]))
                    })

                player_html = build_zero_flicker_player([_render_tracker_html_fixed(s, an) for s in snapshots])
                
                grid_results = []
                for traj in all_trajectories:
                    i, j = traj["node_i"], traj["node_j"]
                    is_dec, is_feas = check_boolean_regions([grids_c[0][c][i, j] for c in c_vars], c_vars, tau_hist[0], {g: grids_g[0][g][i, j] for g in g_vars}, _prob_mod.CONSTRAINTS_LOGIC)
                    grid_results.append({
                        'idx': traj['node_id'], 'is_decoupled': is_dec, 'is_feasible': is_feas,
                        'is_valid': is_dec and is_feas, 'start_norm': traj['start_norm'], 
                        'history': traj['history'], **traj['d_dict'], **traj['opt_res']
                    })
                df_grid = pd.DataFrame(grid_results)

                bests = {'s1': None, 's2': None, 's3': None}
                vs_mask = snapshots[0]['Valid']
                if vs_mask.sum() > 0:
                    v_idx = np.argwhere(vs_mask)
                    f_v, c_v = np.array([grids_obj[0][i, j] for i, j in v_idx]), np.array([an.Coupling_error[i, j] for i, j in v_idx])
                    f_n = (f_v - f_v.min()) / (f_v.max() - f_v.min() + 1e-12)
                    c_n = (c_v - c_v.min()) / (c_v.max() - c_v.min() + 1e-12)
                    
                    for key, attr in zip(['s1', 's2', 's3'], [v_idx[np.argmin(f_v)], v_idx[np.argmin(c_v)], v_idx[np.argmin(np.sqrt(f_n**2 + c_n**2))]]):
                        bests[key] = df_grid[df_grid['idx'] == (attr[0] * m_cols + attr[1])].iloc[0].to_dict()

                # SANITIZE: Save ONLY pure data. No SOM/OpenMDAO objects.
                results_dict[mode]['methods'][method_name] = {
                    'tau_history': tau_hist, 'df_grid': df_grid, 'bests': bests, 'video_player': player_html,
                    'pareto_data': {
                        'v_n': c_v if vs_mask.sum()>0 else [], 'v_o': f_v if vs_mask.sum()>0 else [],
                        's1_c': an.Coupling_error[attr[0], attr[1]] if bests['s1'] else None, 's1_f': grids_obj[0][attr[0], attr[1]] if bests['s1'] else None,
                        's2_c': an.Coupling_error[attr[0], attr[1]] if bests['s2'] else None, 's2_f': grids_obj[0][attr[0], attr[1]] if bests['s2'] else None,
                        's3_c': an.Coupling_error[attr[0], attr[1]] if bests['s3'] else None, 's3_f': grids_obj[0][attr[0], attr[1]] if bests['s3'] else None,
                    }
                }

        st.write("💾 Saving sanitized cache to hard drive for Cloud Deployment...")
        with open(cache_file, "wb") as f:
            pickle.dump(results_dict, f)
            
        status.update(label="✅ Computation Finished & Saved!", state="complete", expanded=False)
    return results_dict


# =============================================================================
# 4. MAIN APP EXECUTION (UI ONLY)
# =============================================================================
def run_dashboard_app(prob_mod):
    st.set_page_config(page_title="IDF Analysis", layout="wide")
    st.markdown("<style>.main { padding-top: 1rem; } .stTabs[data-baseweb='tab-list'] { gap: 8px; } .stTabs[data-baseweb='tab'] { padding: 8px 16px; border-radius: 6px; }</style>", unsafe_allow_html=True)

    st.sidebar.title("⚙️ IDF MDO Dashboard")
    st.sidebar.markdown(f"**Loaded Problem:** `{prob_mod.__name__}`")

    N_SAMPLES = getattr(prob_mod, 'DATASET_SIZE', 50)
    SOM_GRID_SIZE = getattr(prob_mod, 'SOM_GRID_SIZE', (25, 25))

    case_mode = st.sidebar.radio("🧠 Select iSOM Case Mode", [1, 2, 3], index=2)
    tau_method = st.sidebar.radio("📈 Tau Tracking Method", ["Fixed", "Variable", "Floor"], index=0)

    # 1. Fetch Sanitized Data
    all_results = compute_everything_upfront(prob_mod, N_SAMPLES, SOM_GRID_SIZE)
    current_case = all_results[case_mode]
    current_method = current_case['methods'][tau_method]

    tau_history, df_all, bests = current_method['tau_history'], current_method['df_grid'].copy(), current_method['bests']
    pareto_data = current_method['pareto_data']
    tau_0 = tau_history[0]
    c_cols = prob_mod.METADATA["consistency_vars"]

    df_conv = df_all[df_all['converged']].copy()
    GLOBAL_OBJ = df_conv['obj'].min()
    
    df_all['reached_global'] = df_all.apply(lambda r: r['converged'] and abs(r['obj'] - GLOBAL_OBJ) < OBJ_TOL, axis=1)
    df_conv['reached_global'] = df_conv.apply(lambda r: r['converged'] and abs(r['obj'] - GLOBAL_OBJ) < OBJ_TOL, axis=1)
    baseline = df_conv['n_evals'].mean()

    INCLUSIVE_SETS = [
        ("Valid Space (D + F)", df_conv['is_valid'], COL['Valid Space (D + F)']),
        ("Decoupled Only", df_conv['is_decoupled'] & (~df_conv['is_feasible']), COL['Decoupled Only']),
        ("Feasible Only", (~df_conv['is_decoupled']) & df_conv['is_feasible'], COL['Feasible Only']),
        ("Neither", (~df_conv['is_decoupled']) & (~df_conv['is_feasible']), COL['Neither'])
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Initial τ Thresholds ({tau_method}):**")
    for k, v in tau_0.items(): st.sidebar.markdown(f"&nbsp;&nbsp;`{k}`: {v:.4e}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", "❓ Basins", "🔵 Trajectory", "📈 Iters",
        "⚡ Efficiency", "🧮 Metrics", "🎯 Pareto", "🧠 Bokeh Dashboard", "🎬 Video Tracker"
    ])

    with tab1:
        st.header(f"IDF Region Analysis — SOM Grid Performance (Case {case_mode} | {tau_method})")
        valid_evals_mean = df_conv[df_conv['is_valid']]['n_evals'].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Grid Nodes Evaluated", len(df_all))
        c2.metric("Converged Runs", len(df_conv))
        c3.metric("Baseline (Mean evals of whole grid)", f"{baseline:.1f}")

        st.markdown("---")
        vs_col, _ = st.columns([1, 2])
        vs_col.metric("🟢 Mean Evals of Nodes in Valid Space", f"{valid_evals_mean:.1f}", delta=f"{(baseline - valid_evals_mean) / baseline * 100:.1f}% Eval Saving vs Baseline")

        if bests['s3']:
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("🔴 S1 (Min Objective)", bests['s1']['n_evals'], delta=f"{(baseline - bests['s1']['n_evals']) / baseline * 100:.1f}% Eval Saving")
            col_s2.metric("🔵 S2 (Min C-Norm)", bests['s2']['n_evals'], delta=f"{(baseline - bests['s2']['n_evals']) / baseline * 100:.1f}% Eval Saving")
            col_s3.metric("🟠 S3 (Pareto Knee)", bests['s3']['n_evals'], delta=f"{(baseline - bests['s3']['n_evals']) / baseline * 100:.1f}% Eval Saving")

            fig_cmp = go.Figure(go.Bar(
                x=['Entire Grid (Baseline)', 'Valid Space Mean', 'S1 (Min Obj)', 'S2 (Min Norm)', 'S3 (Knee)'],
                y=[baseline, valid_evals_mean, bests['s1']['n_evals'], bests['s2']['n_evals'], bests['s3']['n_evals']],
                marker_color=['gray', COL['Valid Space (D + F)'], '#CB181D', '#1f77b4', '#D94801'],
                text=[f"{v:.1f}" for v in [baseline, valid_evals_mean, bests['s1']['n_evals'], bests['s2']['n_evals'], bests['s3']['n_evals']]], textposition='outside'))
            st.plotly_chart(fig_cmp.update_layout(title="Efficiency Comparison", height=380), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            counts = [df_conv[mask].shape[0] for _, mask, _ in INCLUSIVE_SETS]
            names, colors  = [n for n, _, _ in INCLUSIVE_SETS], [c for _, _, c in INCLUSIVE_SETS]
            fig = go.Figure(go.Bar(x=names, y=counts, marker_color=colors, text=counts, textposition='outside'))
            st.plotly_chart(fig.update_layout(title='Grid Nodes Count by Initial Category', height=320), use_container_width=True)

        with col_r:
            sc = df_all['reached_global'].value_counts()
            fig = go.Figure(go.Bar(x=['Reached Global Optimum', 'Did NOT Reach Global'], y=[sc.get(True, 0), sc.get(False, 0)], text=[sc.get(True, 0), sc.get(False, 0)], textposition='outside', marker_color=['green', 'red']))
            st.plotly_chart(fig.update_layout(title='Global Optimum Success Count', height=320), use_container_width=True)

        st.subheader("Complete Summary Scorecard")
        rows_tbl = []
        for name, mask, _ in INCLUSIVE_SETS:
            sub_all, sub_conv = df_all[mask], df_conv[mask]
            if sub_all.empty: continue
            me = sub_conv['n_evals'].mean() if not sub_conv.empty else np.nan
            rows_tbl.append({
                'Category': name, 'Data Points Count': len(sub_all),
                'Mean evals': f"{me:.1f}" if pd.notna(me) else "N/A",
                'vs. grid baseline': f"{(baseline-me)/baseline*100:+.1f}%" if pd.notna(me) else "N/A",
                '% Reached Global Optimum': f"{sub_all['reached_global'].sum()/len(sub_all)*100:.0f}%"
            })
            
        if bests['s3']:
            for anchor_name, anchor_key in [("S1 (Min Obj)", "s1"), ("S2 (Min C-Norm)", "s2"), ("S3 (Pareto Knee)", "s3")]:
                pt = bests[anchor_key]
                is_global = pt.get('converged', False) and abs(pt.get('obj', float('inf')) - GLOBAL_OBJ) < OBJ_TOL
                rows_tbl.append({
                    'Category': f"⭐ {anchor_name}", 'Data Points Count': 1,
                    'Mean evals': f"{pt['n_evals']:.1f}", 'vs. grid baseline': f"{(baseline-pt['n_evals'])/baseline*100:+.1f}%",
                    '% Reached Global Optimum': "100%" if is_global else "0%"
                })
        st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

    with tab2:
        st.header("Convergence Basins")
        col_l, col_r = st.columns(2)
        with col_l:
            if len(c_cols) >= 2:
                fig = go.Figure()
                for name, mask, color in INCLUSIVE_SETS:
                    sub = df_conv[mask]
                    if not sub.empty: fig.add_trace(go.Scatter(x=[r['history'][-1][c_cols[0]] for _, r in sub.iterrows()], y=[r['history'][-1][c_cols[1]] for _, r in sub.iterrows()], mode='markers', name=name, marker=dict(color=color, opacity=0.6)))
                fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Optimum (0,0)', marker=dict(color='black', size=16, symbol='star')))
                st.plotly_chart(fig.update_layout(title=f'Final ({c_cols[0]},{c_cols[1]})', height=420), use_container_width=True)
        with col_r:
            fig = go.Figure()
            for success in [True, False]:
                sub = df_all[df_all['reached_global'] == success]
                fig.add_trace(go.Scatter(x=sub['start_norm'], y=sub['n_evals'], mode='markers', name='Reached Global' if success else 'Not Global', marker=dict(size=6, opacity=0.6)))
            if bests['s3']: fig.add_trace(go.Scatter(x=[bests['s3']['start_norm']], y=[bests['s3']['n_evals']], mode='markers', name='iSOM Knee (S3)', marker=dict(color='black', size=16, symbol='star')))
            for k, v in tau_0.items(): fig.add_vline(x=v, line_dash='dot', line_color='orange', annotation_text=f'τ_{k}={v:.3f}', annotation_position='top right')
            st.plotly_chart(fig.update_layout(title='Starting ‖c‖₂ vs evals', height=420), use_container_width=True)

    with tab3:
        st.header(f"Dynamic Trajectory Analysis ({tau_method} Tau)")
        c1_ctrl, c2_ctrl = st.columns([3, 1])
        opts = ['iSOM S1 (Min-Obj)', 'iSOM S2 (Min-Norm)', 'iSOM S3 (Pareto Knee)', 'Random from Valid Space', 'Completely Random Grid Node', 'Custom Manual Entry']
        chosen = c1_ctrl.selectbox("Select start:", opts)
        
        custom_pt = {}
        if chosen == 'Custom Manual Entry':
            cols = st.columns(len(prob_mod.METADATA['design_vars']))
            for idx, var in enumerate(prob_mod.METADATA['design_vars']):
                custom_pt[var] = cols[idx].number_input(f"{var}", value=float(df_all[var].mean()), format="%.4f")

        if c2_ctrl.button("▶ Run Optimizer Now", type="primary", use_container_width=True):
            pt = None
            if chosen == 'iSOM S1 (Min-Obj)' and bests['s1']: pt = bests['s1']
            elif chosen == 'iSOM S2 (Min-Norm)' and bests['s2']: pt = bests['s2']
            elif chosen == 'iSOM S3 (Pareto Knee)' and bests['s3']: pt = bests['s3']
            elif 'Valid Space' in chosen: pt = df_all[df_all['is_valid']].sample(1).iloc[0].to_dict()
            elif chosen == 'Custom Manual Entry': pt = custom_pt
            else: pt = df_all.sample(1).iloc[0].to_dict()

            with st.spinner("Running Optimization dynamically..."):
                # Note: This is the ONLY place openmdao runs during Streamlit online. 
                # This single run takes ~0.5 seconds, which is perfectly safe for Streamlit Cloud!
                live_res = prob_mod.run_optimizer({k: pt[k] for k in prob_mod.METADATA['design_vars']})

            df_hist = pd.DataFrame(live_res['history'])
            df_hist['Iteration'] = range(1, len(df_hist) + 1)

            bg_colors, max_t = [], len(tau_history) - 1
            for t_idx, row in df_hist.iterrows():
                t = min(t_idx, max_t)
                phys_out = prob_mod.evaluate_sellar_physics({k: row[k] for k in prob_mod.METADATA['design_vars']})
                is_dec, is_feas = check_boolean_regions([phys_out[c] for c in c_cols], c_cols, tau_history[t], {g: phys_out[g] for g in prob_mod.METADATA['constraint_vars']}, prob_mod.CONSTRAINTS_LOGIC)
                
                if is_dec and is_feas: bg_col = COL['Valid Space (D + F)']
                elif is_dec:           bg_col = COL['Decoupled Only']
                elif is_feas:          bg_col = COL['Feasible Only']
                else:                  bg_col = COL['Neither']
                bg_colors.append(bg_col)

            df_hist['BG_Color'] = bg_colors

            if abs(live_res['obj'] - GLOBAL_OBJ) < OBJ_TOL: st.success(f"Converged to GLOBAL optimum in {live_res['n_evals']} iterations! Obj = {live_res['obj']:.4f}")
            else: st.warning(f"Converged to LOCAL solution in {live_res['n_evals']} iterations. Obj = {live_res['obj']:.4f}")

            def add_bg(fig_obj):
                for i in range(len(df_hist) - 1): fig_obj.add_vrect(x0=df_hist['Iteration'].iloc[i], x1=df_hist['Iteration'].iloc[i+1], fillcolor=df_hist['BG_Color'].iloc[i], opacity=0.15, layer='below', line_width=0)
                return fig_obj

            rc1, rc2 = st.columns(2)
            fig1 = px.line(df_hist, x='Iteration', y=[v for v in prob_mod.METADATA['design_vars'] if v in df_hist.columns], markers=True, title="Design Variables")
            rc1.plotly_chart(add_bg(fig1), use_container_width=True)

            fig2 = px.line(df_hist, x='Iteration', y=[v for v in c_cols if v in df_hist.columns], log_y=True, markers=True, title="Consistency Error")
            for k, v in tau_0.items(): fig2.add_hline(y=v, line_dash='dash', line_color='orange', annotation_text=f'Initial τ_{k}={v:.3f}')
            rc2.plotly_chart(add_bg(fig2), use_container_width=True)

            if len(c_cols) >= 2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=[-tau_0[c_cols[0]], tau_0[c_cols[0]], tau_0[c_cols[0]], -tau_0[c_cols[0]], -tau_0[c_cols[0]]], y=[-tau_0[c_cols[1]], -tau_0[c_cols[1]], tau_0[c_cols[1]], tau_0[c_cols[1]], -tau_0[c_cols[1]]], mode='lines', line=dict(color='orange', dash='dash'), name='Initial Boundary'))
                fig4.add_trace(go.Scatter(x=df_hist[c_cols[0]], y=df_hist[c_cols[1]], mode='lines', line=dict(color='gray', width=2), name='Path'))
                for color in df_hist['BG_Color'].unique():
                    name_map = {v: k for k, v in COL.items()}
                    fig4.add_trace(go.Scatter(x=df_hist[df_hist['BG_Color'] == color][c_cols[0]], y=df_hist[df_hist['BG_Color'] == color][c_cols[1]], mode='markers', marker=dict(color=color, size=8), name=name_map.get(color, "Step")))
                st.plotly_chart(fig4.update_layout(title=f"Phase Portrait ({c_cols[0]} vs {c_cols[1]})", height=450), use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        fig_box = go.Figure()
        for name, mask, color in INCLUSIVE_SETS: 
            if not df_conv[mask].empty: fig_box.add_trace(go.Box(y=df_conv[mask]['n_evals'], name=name, marker_color=color))
        if bests['s1']: fig_box.add_hline(y=bests['s1']['n_evals'], line_dash='dash', line_color='#CB181D', annotation_text='S1 (Min Obj)')
        if bests['s2']: fig_box.add_hline(y=bests['s2']['n_evals'], line_dash='dash', line_color='#1f77b4', annotation_text='S2 (Min C-Norm)')
        if bests['s3']: fig_box.add_hline(y=bests['s3']['n_evals'], line_dash='dash', line_color='#D94801', annotation_text='S3 (Knee)')
        if not np.isnan(valid_evals_mean): fig_box.add_hline(y=valid_evals_mean, line_dash='dot', line_color='green', annotation_text='Valid Space Mean')
        c1.plotly_chart(fig_box.update_layout(title='Evals by region', height=400), use_container_width=True)
        
        fig_cdf = go.Figure()
        for name, mask, color in INCLUSIVE_SETS:
            ev = np.sort(df_conv[mask]['n_evals'].values)
            if len(ev) > 0: fig_cdf.add_trace(go.Scatter(x=ev, y=np.arange(1, len(ev)+1)/len(ev), mode='lines', name=name, line=dict(color=color, width=2)))
        c2.plotly_chart(fig_cdf.update_layout(title='CDF of Evals', height=400), use_container_width=True)

    with tab5:
        st.header("Efficiency of starting regions")
        col_l, col_r = st.columns(2)
        with col_l:
            fig_bar = go.Figure(go.Bar(
                x=[name for name, mask, color in INCLUSIVE_SETS if not df_conv[mask].empty],
                y=[(baseline - df_conv[mask]['n_evals'].mean())/baseline*100 for name, mask, color in INCLUSIVE_SETS if not df_conv[mask].empty],
                marker_color=[color for name, mask, color in INCLUSIVE_SETS if not df_conv[mask].empty],
                text=[f"{(baseline - df_conv[mask]['n_evals'].mean())/baseline*100:+.1f}%" for name, mask, color in INCLUSIVE_SETS if not df_conv[mask].empty], textposition='outside'))
            st.plotly_chart(fig_bar.update_layout(title='% Eval Reduction vs Mean', yaxis_title='% reduction', height=400), use_container_width=True)
        with col_r:
            fig_vio = go.Figure()
            for name, mask, color in INCLUSIVE_SETS:
                ev = df_conv[mask]['n_evals']
                if not ev.empty: fig_vio.add_trace(go.Violin(y=ev, name=name, fillcolor=color, line_color=color))
            if bests['s3']: fig_vio.add_hline(y=bests['s3']['n_evals'], line_dash='dash', line_color='black')
            st.plotly_chart(fig_vio.update_layout(title='Distribution of evaluations', height=400, showlegend=False), use_container_width=True)

    with tab6:
        st.header(f"Region Stability Metrics ({tau_method} Tau)")
        rows, max_t = [], len(tau_history) - 1
        for _, row in df_conv.iterrows():
            if not row['history']: continue
            dec_mask = [all(abs(h[c]) < tau_history[min(t_idx, max_t)][c] for c in c_cols if c in h) for t_idx, h in enumerate(row['history'])]
            if not dec_mask: continue
            rows.append({'is_decoupled': row['is_decoupled'], 'is_feasible': row['is_feasible'], 'retained': dec_mask[0], 'frac_dec': sum(dec_mask)/len(dec_mask)})
        mdf = pd.DataFrame(rows)

        col_l, col_r = st.columns(2)
        with col_l:
            if not mdf.empty:
                r_labs, r_vals, r_cols = [], [], []
                for name, mask, color in INCLUSIVE_SETS:
                    sub_m = mdf[mdf['is_decoupled'] & mdf['is_feasible']] if "Valid" in name else mdf[mdf['is_decoupled'] & (~mdf['is_feasible'])] if "Decoupled" in name else mdf[(~mdf['is_decoupled']) & mdf['is_feasible']] if "Feasible" in name else mdf[(~mdf['is_decoupled']) & (~mdf['is_feasible'])]
                    if not sub_m.empty: r_labs.append(name); r_vals.append(sub_m['retained'].mean()*100); r_cols.append(color)
                st.plotly_chart(go.Figure(go.Bar(x=r_labs, y=r_vals, marker_color=r_cols, text=[f'{v:.1f}%' for v in r_vals], textposition='outside')).update_layout(title='Iteration 1 Retention', height=380), use_container_width=True)
        with col_r:
            if not mdf.empty:
                fig = go.Figure()
                for name, mask, color in INCLUSIVE_SETS:
                    sub_m = mdf[mdf['is_decoupled'] & mdf['is_feasible']] if "Valid" in name else mdf[mdf['is_decoupled'] & (~mdf['is_feasible'])] if "Decoupled" in name else mdf[(~mdf['is_decoupled']) & mdf['is_feasible']] if "Feasible" in name else mdf[(~mdf['is_decoupled']) & (~mdf['is_feasible'])]
                    if not sub_m.empty: fig.add_trace(go.Violin(y=sub_m['frac_dec'], name=name, fillcolor=color, line_color=color))
                st.plotly_chart(fig.update_layout(title='Fraction in decoupled region', height=380, showlegend=False), use_container_width=True)

    with tab7:
        st.header("iSOM Pareto Front (Extracted from Pure Data Cache)")
        if len(pareto_data['v_n']) > 0:
            v_n, v_o = pareto_data['v_n'], pareto_data['v_o']
            n_pts, pareto_mask = len(v_n), np.ones(len(v_n), dtype=bool)
            for p in range(n_pts):
                for q in range(n_pts):
                    if p != q and (v_n[q] <= v_n[p] and v_o[q] <= v_o[p] and (v_n[q] < v_n[p] or v_o[q] < v_o[p])):
                        pareto_mask[p] = False; break
            pn, po = v_n[pareto_mask], v_o[pareto_mask]
            sorder = np.argsort(pn)
            pn, po = pn[sorder], po[sorder]

            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(x=v_n, y=v_o, mode='markers', marker=dict(color='lightgray'), name="Valid space points"))
            fig_pf.add_trace(go.Scatter(x=pn, y=po, mode='lines+markers', line=dict(color='#2171B5', width=2), marker=dict(color='#2171B5', size=8), name=f"Pareto front ({len(pn)} pts)"))
            
            if pareto_data['s1_f'] is not None: fig_pf.add_trace(go.Scatter(x=[pareto_data['s1_c']], y=[pareto_data['s1_f']], mode='markers', marker=dict(color='#CB181D', size=18, symbol='star'), name="S1 (Min Obj)"))
            if pareto_data['s2_f'] is not None: fig_pf.add_trace(go.Scatter(x=[pareto_data['s2_c']], y=[pareto_data['s2_f']], mode='markers', marker=dict(color='#1f77b4', size=18, symbol='star'), name="S2 (Min C-Norm)"))
            if pareto_data['s3_f'] is not None: fig_pf.add_trace(go.Scatter(x=[pareto_data['s3_c']], y=[pareto_data['s3_f']], mode='markers', marker=dict(color='#D94801', size=20, symbol='diamond'), name="S3 (Pareto Knee SELECTED)"))
            st.plotly_chart(fig_pf.update_layout(xaxis_title="C-Norm", yaxis_title="Objective", height=500), use_container_width=True)

    with tab8:
        st.header(f"🧠 Full iSOM Bokeh Dashboard (Case Mode {case_mode})")
        components.html(current_case['bokeh_html_string'], height=1400, scrolling=True)

    with tab9:
        st.header(f"🎬 Dynamic Video Player (Case {case_mode} | {tau_method})")
        components.html(current_method['video_player'], height=600, scrolling=False)