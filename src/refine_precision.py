# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import jax
# import jax.experimental.optimizers
# --- JAX optimizer compatibility patch ---
# try:
#     import jax.experimental.optimizers as optimizers
# except ImportError:
#     import optax
#     class OptimizerWrapper:
#         # mimic old API: init_fun, update_fun, get_params
#         def __init__(self, step_size):
#             self.opt = optax.sgd(step_size)
#         def init_fun(self, params):
#             return self.opt.init(params)
#         def update_fun(self, grad, state, params):
#             updates, new_state = self.opt.update(grad, state, params)
#             new_params = optax.apply_updates(params, updates)
#             return new_params, new_state
#         def get_params(self, state):
#             return state

#     def make_optimizer(step_size):
#         return OptimizerWrapper(step_size)
#     optimizers = type("optimizers", (), {"sgd": make_optimizer})
# --- Optimizer compatibility shim for old jax.experimental.optimizers.adam API ---
# This provides `adam(lr)` -> (init_fn, opt_update_fn, get_params_fn)


import jax.numpy as jnp

from src.global_vars import *
from src.utils import matmul, which_is_zero
from src.find_witnesses import do_better_sweep
# ----------------- JAX optimizer compatibility shim -----------------
# Place this near the top of src/refine_precision.py (replace any previous shim)
import jax

try:
    # Prefer the legacy module if present
    import jax.experimental.optimizers as _jax_optimizers
    optimizers = _jax_optimizers
    # ensure jax.experimental has optimizers attribute
    try:
        if not hasattr(jax, "experimental") or jax.experimental is None:
            import types
            jax.experimental = types.SimpleNamespace()
        jax.experimental.optimizers = _jax_optimizers
    except Exception:
        pass
    _OPT_BACKEND = "jax.experimental.optimizers"
except Exception:
    # Fallback to optax and emulate the old API: adam(lr) -> (init_fn, opt_update_fn, get_params_fn)
    try:
        import optax
    except Exception as e:
        raise ImportError("Neither jax.experimental.optimizers nor optax is available. Install optax with `python3 -m pip install optax`.") from e

    def adam(step_size):
        """
        Emulate old jax.experimental.optimizers.adam API using optax.
        Returns: init_fn(params), opt_update_fn(step_index, grads, state), get_params_fn(state)
        State format here: (params, opt_state)
        """
        tx = optax.adam(step_size)

        def init_fn(params):
            opt_state = tx.init(params)
            return (params, opt_state)

        def opt_update_fn(i, grads, state):
            params, opt_state = state
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state)

        def get_params_fn(state):
            return state[0]

        return init_fn, opt_update_fn, get_params_fn

    # expose as optimizers.adam and also attach to jax.experimental.optimizers so old import paths work
    class _Shim:
        @staticmethod
        def adam(step_size):
            return adam(step_size)

    optimizers = _Shim()
    try:
        import types
        if not hasattr(jax, "experimental") or jax.experimental is None:
            jax.experimental = types.SimpleNamespace()
        jax.experimental.optimizers = optimizers
    except Exception:
        # non-fatal if we can't attach, but most callers will use the optimizers symbol we provide
        pass

    _OPT_BACKEND = "optax (shim)"

# Optional: print backend for debug
print("Optimizer backend for refine_precision:", _OPT_BACKEND)
# -------------------------------------------------------------------

def trim(hidden_layer, out, num_good):
    """
    Compute least squares in a robust-statistics manner.
    See Jagielski et al. 2018 S&P
    """
    lst, *rest = np.linalg.lstsq(hidden_layer, out)
    old = lst
    for _ in range(20):
        errs = np.abs(np.dot(hidden_layer, lst) - out)
        best_errs = np.argsort(errs)[:num_good]
        lst, *rest = np.linalg.lstsq(hidden_layer[best_errs], out[best_errs])
        if np.linalg.norm(old-lst) < 1e-9:
            return lst, best_errs
        old = lst
    return lst, best_errs

def improve_row_precision(args):
    """
    Improve the precision of an extracted row.
    We think we know where it is, but let's actually figure it out for sure.

    To do this, start by sampling a bunch of points near where we expect the line to be.
    This gives us a picture like this

                      X
                       X
                    
                   X
               X
                 X
                X

    Where some are correct and some are wrong.
    With some robust statistics, try to fit a line that fits through most of the points
    (in high dimension!)

                      X
                     / X
                    / 
                   X
               X  /
                 /
                X

    This solves the equation and improves the point for us.
    """
    (LAYER, known_T, known_A, known_B, row, did_again) = args
    print("Improve the extracted neuron number", row)

    print(np.sum(np.abs(known_A[:,row])))
    if np.sum(np.abs(known_A[:,row])) < 1e-8:
        return known_A[:,row], known_B[row]
        

    def loss(x, r):
        hidden = known_T.forward(x, with_relu=True, np=jnp)
        dotted = matmul(hidden, jnp.array(known_A)[:,r], jnp.array(known_B)[r], np=jnp)
                
        return jnp.sum(jnp.square(dotted))

    loss_grad = jax.jit(jax.grad(loss))
    loss = jax.jit(loss)

    extended_T = known_T.extend_by(known_A, known_B)

    def get_more_points(NUM):
        """
        Gather more points. This procedure is really kind of ugly and should probably be fixed.
        We want to find points that are near where we expect them to be.

        So begin by finding preimages to points that are on the line with gradient descent.
        This should be completely possible, because we have d_0 input dimensions but 
        only want to control one inner layer.
        """
        print("Gather some more actual critical points on the plane")
        stepsize = .1
        critical_points = []
        while len(critical_points) <= NUM:
            print("On this iteration I have ", len(critical_points), "critical points on the plane")
            points = np.random.normal(0, 1e3, size=(100,DIM,))
            
            lr = 10
            for step in range(5000):
                # Use JaX's built in optimizer to do this.
                # We want to adjust the LR so that we get a better solution
                # as we optimize. Probably there is a better way to do this,
                # but this seems to work just fine.

                # No queries involvd here.
                if step%1000 == 0:
                    lr *= .5
                    init, opt_update, get_params = jax.experimental.optimizers.adam(lr)
                
                    @jax.jit
                    def update(i, opt_state, batch):
                        params = get_params(opt_state)
                        return opt_update(i, loss_grad(batch, row), opt_state)
                    opt_state = init(points)
                
                if step%100 == 0:
                    ell = loss(points, row)
                    if CHEATING:
                        # This isn't cheating, but makes things prettier
                        print(ell)
                    if ell < 1e-5:
                        break
                opt_state = update(step, opt_state, points)
                # points = opt_state.packed_state[0][0]
                # opt_state format under shim = (params, opt_state)
                try:
                    # old jax.experimental.optimizers
                    points = opt_state.packed_state[0][0]
                except AttributeError:
                    # new optax-shim path
                    points = opt_state[0]  # params

                
            for point in points:
                # For each point, try to see where it actually is.

                # First, if optimization failed, then abort.
                if loss(point, row) > 1e-5:
                    continue

                if LAYER > 0:
                    # If wee're on a deeper layer, and if a prior layer is zero, then abort
                    if min(np.min(np.abs(x)) for x in known_T.get_hidden_layers(point)) < 1e-4:
                        print("is on prior")
                        continue
                    
                    
                #print("Stepsize", stepsize)
                tmp = query_count
                solution = do_better_sweep(offset=point,
                                           low=-stepsize,
                                           high=stepsize,
                                           known_T=known_T)
                #print("qs", query_count-tmp)
                if len(solution) == 0:
                    stepsize *= 1.1
                elif len(solution) > 1:
                    stepsize /= 2
                elif len(solution) == 1:
                    stepsize *= 0.98
                    potential_solution = solution[0]

                    hiddens = extended_T.get_hidden_layers(potential_solution)


                    this_hidden_vec = extended_T.forward(potential_solution)
                    this_hidden = np.min(np.abs(this_hidden_vec))
                    if min(np.min(np.abs(x)) for x in this_hidden_vec) > np.abs(this_hidden)*0.9:
                        critical_points.append(potential_solution)
                    else:
                        print("Reject it")
        print("Finished with a total of", len(critical_points), "critical points")
        return critical_points


    critical_points_list = []
    for _ in range(1):
        NUM = sizes[LAYER]*2
        critical_points_list.extend(get_more_points(NUM))
        
        critical_points = np.array(critical_points_list)

        hidden_layer = known_T.forward(np.array(critical_points), with_relu=True)

        if CHEATING:
            out = np.abs(matmul(hidden_layer, A[LAYER],B[LAYER]))
            which_neuron = int(np.median(which_is_zero(0, [out])))
            print("NEURON NUM", which_neuron)

            crit_val_0 = out[:,which_neuron]
                
            print(crit_val_0)

            #print(list(np.sort(np.abs(crit_val_0))))
            print('probability ok',np.mean(np.abs(crit_val_0)<1e-8))

        crit_val_1 = matmul(hidden_layer, known_A[:,row], known_B[row])

        best = (None, 1e6)
        upto = 100

        for iteration in range(upto):
            if iteration%1000 == 0:
                print("ITERATION", iteration, "OF", upto)
            if iteration%2 == 0 or True:

                # Try 1000 times to make sure that we get at least one non-zero per axis
                for _ in range(1000):
                    randn = np.random.choice(len(hidden_layer), NUM+2, replace=False)
                    if np.all(np.any(hidden_layer[randn] != 0, axis=0)):
                        break

                hidden = hidden_layer[randn]
                soln,*rest = np.linalg.lstsq(hidden, np.ones(hidden.shape[0]))
                
                
            else:
                randn = np.random.choice(len(hidden_layer), min(len(hidden_layer),hidden_layer.shape[1]+20), replace=False)
                soln,_ = trim(hidden_layer[randn], np.ones(hidden_layer.shape[0])[randn], hidden_layer.shape[1])


            crit_val_2 = matmul(hidden_layer, soln, None)-1
            
            quality = np.median(np.abs(crit_val_2))

            if iteration%100 == 0:
                print('quality', quality, best[1])
            
            if quality < best[1]:
                best = (soln, quality)

            if quality < 1e-10: break
            if quality < 1e-10 and iteration > 1e4: break
            if quality < 1e-8 and iteration > 1e5: break

        soln, _ = best

        if CHEATING:
            print("Compare", np.median(np.abs(crit_val_0)))
        print("Compare",
              np.median(np.abs(crit_val_1)),
              best[1])

        if np.all(np.abs(soln) > 1e-10):
            break

    print('soln',soln)
    
    if np.any(np.abs(soln) < 1e-10):
        print("THIS IS BAD. FIX ME NOW.")
        exit(1)
    
    rescale = np.median(soln/known_A[:,row])
    soln[np.abs(soln) < 1e-10] = known_A[:,row][np.abs(soln) < 1e-10] * rescale

    if CHEATING:
        other = A[LAYER][:,which_neuron]
        print("real / mine / diff")
        print(other/other[0])
        print(soln/soln[0])
        print(known_A[:,row]/known_A[:,row][0])
        print(other/other[0] - soln/soln[0])

    
    if best[1] < np.mean(np.abs(crit_val_1)) or True:
        return soln, -1
    else:
        print("FAILED TO IMPROVE ACCURACY OF ROW", row)
        print(np.mean(np.abs(crit_val_2)), 'vs', np.mean(np.abs(crit_val_1)))
        return known_A[:,row], known_B[row]


def improve_layer_precision(LAYER, known_T, known_A, known_B):
    new_A = []
    new_B = []    

    out = map(improve_row_precision,
              [(LAYER, known_T, known_A, known_B, row, False) for row in range(neuron_count[LAYER+1])])
    new_A, new_B = zip(*out)

    new_A = np.array(new_A).T
    new_B = np.array(new_B)

    print("HAVE", new_A, new_B)

    return new_A, new_B
