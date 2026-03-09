# 2D Rotated GPE Benchmark

## Folder structure

- **YK_long**  
  ITE method used in the paper, which is an **explicit finite element imaginary time method**.

- **FD**  
  **Backward Euler Finite Difference (BEFD)** method.

- **SP**  
  **Backward Euler Pseudo-Spectral (BESP)** method.

The BEFD and BESP schemes follow the implementations described in the **GPELab toolbox**:  
https://gpelab.math.cnrs.fr/

GPELab provides numerical solvers for Gross–Pitaevskii equations including BEFD and BESP discretizations.

---

## Initial condition

`im.zip` contains the initial wave function with **7 vortices**.

Extract it and you will obtain

```
wave_real_imag.txt
```

Put this file into:

```
FD/Examples
SP/Examples
YK_long
```

---

## Usage

First extract the initial condition:

```bash
unzip im.zip
```

---

### YK_long

Go to the `YK_long` folder and run

```bash
nohup matlab -nodisplay -nosplash < gpe_run.m > run.log &
```

---

### FD (BEFD)

Go to the `FD` folder and run

```bash
nohup matlab -batch "addpath(genpath(pwd)); addpath(genpath(fullfile(pwd,'Variables'))); BEFD" > run.log 2>&1 &
```

---

### SP (BESP)

Go to the `SP` folder and run

```bash
nohup matlab -batch "addpath(genpath(pwd)); addpath(genpath(fullfile(pwd,'Variables'))); BESP" > run.log 2>&1 &
```

---

## Output

The following quantities will be stored in `.mat` files:

- CPU time
- Energy
- Chemical potential

---

---

## Plotting

Put all `.mat` files in `GD` folder

Also you can run GD method to get QTT GD data by

```bash
nohup python -O GP_vortex_test.py &
```

and we can use `figure.py` to plot all the data.

```bash
python figure.py
```

---

## Reference

The BEFD and BESP schemes follow the numerical framework described in:

X. Antoine, R. Duboscq  
*GPELab, a Matlab Toolbox to Solve Gross–Pitaevskii Equations I: Computation of Stationary Solutions*  
Computer Physics Communications 185 (2014), 2969–2991.

X. Antoine, R. Duboscq  
*GPELab, a Matlab Toolbox to Solve Gross–Pitaevskii Equations II: Dynamics and Stochastic Simulations*  
Computer Physics Communications 193 (2015), 95–117.

More information about the toolbox can be found at:  
https://gpelab.math.cnrs.fr/
