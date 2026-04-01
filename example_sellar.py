"""
Example Sellar Problem Definition
"""
import os, sys, warnings
os.environ["OPENMDAO_REPORTS"] = "0"
os.environ['OPENMDAO_REQUIRE_MPI'] = 'false'
os.environ['OPENMDAO_USE_MPI'] = '0'
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='openmdao')

import numpy as np
import pandas as pd
from scipy.stats import qmc
import openmdao.api as om
import time

# ============================================================
# 1. METADATA & CONFIGURATION (Now controls the Dashboard!)
# ============================================================
DATASET_SIZE = 5
SOM_GRID_SIZE = (5, 5)
TRAIN_ROUGH_LEN = 5
TRAIN_FINETUNE_LEN = 5
METADATA = {
    "design_vars":["z1", "z2", "x1", "y1_hat", "y2_hat"],
    "consistency_vars": ["C1", "C2"],
    "constraint_vars":["g1", "g2"],
    "objective_var": "Objective"
}
CONSTRAINTS_LOGIC = {"g1": (">=", 3.16), "g2": ("<=", 24.0)}

# ============================================================
# 2. PHYSICS & DATASET GENERATION
# ============================================================
def evaluate_sellar_physics(d):
    y1 = d["z1"]**2 + d["z2"] + d["x1"] - 0.2 * d["y2_hat"]
    y2 = np.sqrt(np.abs(d["y1_hat"])) + d["z1"] + d["z2"]
    return {
        "C1": y1 - d["y1_hat"], "C2": y2 - d["y2_hat"],
        "g1": y1, "g2": y2,
        "Objective": d["x1"]**2 + d["z2"] + y1 + np.exp(-y2)
    }

def generate_full_dataset(N=DATASET_SIZE, seed=42):
    U = qmc.LatinHypercube(d=5, seed=seed).random(N)
    d_dict = {
        "z1": -8 + 16 * U[:, 0], "z2": 0 + 15 * U[:, 1], "x1": 0 + 5 * U[:, 2],
        "y1_hat": 0.1 + 49.9 * U[:, 3], "y2_hat": 0.1 + 29.9 * U[:, 4]
    }
    out_dict = evaluate_sellar_physics(d_dict)
    return pd.DataFrame({**d_dict, **out_dict})

# ============================================================
# 3. OPENMDAO OPTIMIZER TRACKING
# ============================================================
class SellarDisc1(om.ExplicitComponent):
    def setup(self):
        for v in['z1', 'z2', 'x1', 'y2_hat']: self.add_input(v, val=1.)
        self.add_output('y1', val=1.)
    def setup_partials(self): self.declare_partials('*', '*', method='fd')
    def compute(self, inputs, outputs):
        outputs['y1'] = inputs['z1'][0]**2 + inputs['z2'][0] + inputs['x1'][0] - 0.2 * inputs['y2_hat'][0]

class SellarDisc2(om.ExplicitComponent):
    def setup(self):
        for v in['z1', 'z2', 'y1_hat']: self.add_input(v, val=1.)
        self.add_output('y2', val=1.)
    def setup_partials(self): self.declare_partials('*', '*', method='fd')
    def compute(self, inputs, outputs):
        outputs['y2'] = np.sqrt(np.abs(inputs['y1_hat'][0])) + inputs['z1'][0] + inputs['z2'][0]

class EvalCounter(om.ExplicitComponent):
    def initialize(self): self.options.declare('history', recordable=False)
    def setup(self):
        for v in['x1', 'z1', 'z2', 'y1', 'y2', 'y1_hat', 'y2_hat']: self.add_input(v, val=1.)
        self.add_output('obj', val=0.)
    def setup_partials(self): self.declare_partials('*', '*', method='fd')
    def compute(self, inputs, outputs):
        obj_val = inputs['x1'][0]**2 + inputs['z2'][0] + inputs['y1'][0] + np.exp(-inputs['y2'][0])
        outputs['obj'] = obj_val
        
        # WE RECORD EVERYTHING FOR THE DYNAMIC DASHBOARD HERE!
        self.options['history'].append({
            'x1': float(inputs['x1'][0]), 'z1': float(inputs['z1'][0]), 'z2': float(inputs['z2'][0]),
            'y1_hat': float(inputs['y1_hat'][0]), 'y2_hat': float(inputs['y2_hat'][0]),
            'C1': float(inputs['y1'][0]-inputs['y1_hat'][0]), 'C2': float(inputs['y2'][0]-inputs['y2_hat'][0]),
            'norm': np.sqrt((inputs['y1'][0]-inputs['y1_hat'][0])**2 + (inputs['y2'][0]-inputs['y2_hat'][0])**2),
            'obj': float(obj_val)
        })

def run_optimizer(start_dict):
    history =[]
    prob = om.Problem(reports=None); model = prob.model
    ivc = om.IndepVarComp()
    for name in METADATA['design_vars']: ivc.add_output(name, start_dict[name])
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('d1', SellarDisc1(), promotes_inputs=['x1','z1','z2','y2_hat'], promotes_outputs=['y1'])
    model.add_subsystem('d2', SellarDisc2(), promotes_inputs=['z1','z2','y1_hat'], promotes_outputs=['y2'])
    model.add_subsystem('cons', om.ExecComp(['C1=y1-y1_hat', 'C2=y2-y2_hat']), promotes=['*'])
    model.add_subsystem('obj_cmp', EvalCounter(history=history), promotes=['*'])
    
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)
    prob.driver.options['disp'] = False
    
    model.add_design_var('x1', lower=0, upper=5)
    model.add_design_var('z1', lower=-8, upper=8)
    model.add_design_var('z2', lower=0, upper=15)
    model.add_design_var('y1_hat', lower=0.1, upper=50)
    model.add_design_var('y2_hat', lower=0.1, upper=30)
    model.add_objective('obj')
    model.add_constraint('y1', lower=3.16); model.add_constraint('y2', upper=24.)
    model.add_constraint('C1', equals=0.); model.add_constraint('C2', equals=0.)
    
    prob.setup(); prob.set_solver_print(level=0)
    t0 = time.perf_counter(); converged = True
    try: prob.run_driver()
    except: converged = False
    t1 = time.perf_counter()
    
    nf = np.sqrt(prob.get_val('C1')[0]**2 + prob.get_val('C2')[0]**2) if converged else np.nan
    return {'n_evals': len(history), 'obj': float(prob.get_val('obj')[0]) if converged else np.nan,
            'converged': converged and nf<1e-4, 'history': history, 'wall_time': t1-t0}

# ============================================================
# 4. EXECUTION ROUTER (BULLETPROOF STREAMLIT CHECK)
# ============================================================
if __name__ == "__main__":
    is_streamlit = False
    if "streamlit" in sys.argv[0].lower() or "streamlit" in sys.modules:
        is_streamlit = True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None: is_streamlit = True
    except ImportError: pass

    if is_streamlit:
        # RUNNING VIA STREAMLIT: "streamlit run example_sellar.py"
        from isom_mdo_dashboard import run_dashboard_app
        # from new import  run_dashboard_app
        run_dashboard_app(sys.modules[__name__])
    else:
        # RUNNING VIA PYTHON: "python example_sellar.py"
        from isom_mdo_scalable import ISOMAnalyzer, run_dashboard
        df_full = generate_full_dataset(N=DATASET_SIZE)
        for mode in[1,2,3]:
            print(f"\n====================================")
            print(f"Running Standalone iSOM - Case {mode}")
            print(f"====================================")
            an = ISOMAnalyzer(df_full, METADATA, mode, SOM_GRID_SIZE)
            an.train(train_rough_len=TRAIN_ROUGH_LEN, train_finetune_len=TRAIN_FINETUNE_LEN)
            an.evaluate(evaluate_sellar_physics)
            an.compute_regions(20, CONSTRAINTS_LOGIC)
            run_dashboard(an, problem_name=f"Sellar_Case_{mode}", show_results=True)
