# Vortex-Lattice Tests (g, Ω) — following arXiv:2507.04279v5

Reference: Q.-C. Chen, I-K. Liu, J.-W. Li, C.-M. Chung,
*Solving the Gross-Pitaevskii Equation with Quantic Tensor Trains: Ground States and Nonlinear Dynamics*,
[arXiv:2507.04279v5](https://arxiv.org/abs/2507.04279v5) (Sec. 4.2, Table 1).

## Model

Dimensionless units (ħ = m = ω = 1), normalization ∫|ψ|² = 1:

$$
\hat H = -\tfrac12\nabla^2 + \tfrac12 r^2 - \Omega \hat L_z + g|\psi|^2 ,\qquad
E[\psi] = \int \psi^*\Big(-\tfrac12\nabla^2 + \tfrac12 r^2 - \Omega \hat L_z\Big)\psi + \frac g2\int|\psi|^4 .
$$

In the QTT code the wave function is normalized on the grid, so the coupling is rescaled as
$g' = g/(\Delta x)^2$ (paper App. A). This is why the scripts set

```python
g = 100/dx**2   # physical g = 100
```

A rough estimate of the vortex number (paper Eq. 10):
$N_v \approx \Omega R_{TF}^2$, with $R_{TF} = \left[\frac{4g}{\pi(1-\Omega^2)}\right]^{1/4}$.

## Test parameters (paper Table 1)

Spatial domain $[-21, 21]^2$, $N_{qub} = 17$ per dimension ($2^{17}\times 2^{17}$ grid) in the paper.
μ and E were obtained with QTT gradient descent (GD).

| Vortices | Ω      | g    | Bond dim D | N_qub | μ    | E    | Saved state |
|---------:|-------:|-----:|-----------:|------:|-----:|-----:|-------------|
| 7        | 0.8    | 100  | 30         | 17    | 4.35 | 3.19 | `Run/7v.pkl`   |
| 19       | 0.946  | 100  | 30         | 17    | 2.90 | 2.25 | `Run/19v.pkl`  |
| 37       | 0.913  | 500  | 40         | 17    | 6.36 | 4.54 | `Run/37v.pkl`  |
| 61       | 0.9616 | 500  | 50         | 17    | 4.67 | 3.42 | `Run/61v.pkl`  |
| 91       | 0.9617 | 1100 | 65         | 17    | 6.46 | 4.59 | `Run/91v.pkl`  |
| 125      | 0.972  | 1550 | 80         | 17    | 6.63 | 4.68 | `Run/125v.pkl` |

Other settings used in the paper / scripts:

- Truncation cutoff for $|\psi|^2$: $\epsilon^2 = 10^{-8}$ (`cutoff_mps2 = 1e-8`).
- Step size: $d\tau = \Delta x^2/2$ (`dt = dx**2/2`).
- The 7v test in Fig. 6 of the paper uses a smaller domain $[-6, 6]^2$ with $N_{qub}=8$.
- The 19v benchmark against GPELab (BESP / BEFD, `paper_compare.py`) uses a $2^{11}\times 2^{11}$ grid on $[-21,21]^2$,
  starting from the 7v state; each data point corresponds to 20 iterations.

## How to run a test

Edit the parameter block in `Run/GP_vortex_test.py` for the target row of the table, e.g. for 37 vortices:

```python
N = 17                 # qubits per dimension (paper uses up to 17)
x1, x2 = -21, 21
dx = (x2 - x1) / 2**N
g = 500/dx**2          # physical g = 500
omega = 0.913
maxdim = 40            # bond dimension D
cutoff_mps2 = 1e-8
steps = 10000
```

and choose the initial state (previous lower-vortex state), e.g.

```python
psi = get_init_other_state(N, x1, x2, maxdim, "19v.pkl")
```

Then run in the background:

```
nohup python -O GP_vortex_test.py &
```

Use `measure_energy_pkl.py` to measure μ and E of a saved `.pkl` state and compare with the table above,
and `plot2D_sec.py` to plot the density $|\psi|^2$ and count the vortices.

## Resolution check (paper App. D.3)

Starting from the 125v state on the $2^{17}\times2^{17}$ grid and coarse-graining to $2^8$–$2^{11}$ points per dimension
(fixed g, Ω, D = 80) converges to *different* vortex numbers — a sufficiently fine grid is required for the large-vortex tests.
