## Workflow for Generating Multi-Vortex States

1. Use **TCI** to fit a **single-vortex state**, which is used as the initial state for generating the **7-vortex (7v)** configuration.

2. Using the paper parameters for **Ω (Omega)**, **g**, and **bond dimension**, run **1000 steps** to obtain the converged **7v state**.

3. Repeat the above procedure to generate higher-vortex states: **19v**, **37v**, and **61v**.

   - For states with a larger number of vortices, more evolution steps are required, and the **mesh grid must be enlarged** with additional perturbations applied to the state.
   - For example, the **61v state** is difficult to obtain directly on a \(2^9 \times 2^9\) mesh. A practical approach is:
     1. Place the **37v state** on a \(2^9 \times 2^9\) mesh and evolve it at **Ω = 0.978** for **600 steps** to obtain an intermediate **61v state**.
     2. Use this intermediate state as the initial state, enlarge the mesh to \(2^{11} \times 2^{11}\), reduce **Ω to 0.9616**, and continue evolving for **200 additional steps**.

