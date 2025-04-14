# 2D GPE Solver with Tensor Network Methods

This repository contains implementations of tensor network-based methods for solving the two-dimensional Gross–Pitaevskii equation (GPE), 
including gradient descent (GD), imaginary time evolution (TDVP), and real-time evolution, all applied to vortex-state dynamics. The codes 
are organized into four main directories:

## Dependencies

The following Python packages are required:

- [xfac](https://github.com/tensor4all/xfac)  
  Please install the **Python version** of `xfac` by following the instructions in the linked repository. Be sure to include the build flag:  
  ```
  -D XFAC_BUILD_PYTHON=ON
  ```
  After installation, you need to linked the package directory in `tci.py`
  ```
  import os, sys 
  sys.path.append('your_dir/xfac/build/python/')
  ```
- numpy  
- scipy  
- matplotlib  
- ncon  
- pickle

Install them using pip:
```
pip3 install numpy scipy matplotlib ncon pickle
```

## Folder Structure

### 1. `GD/`: Gradient Descent Solver

This folder contains a gradient descent algorithm for solving the 2D Gross–Pitaevskii equation.

Run the main script:
```
python -O GP_vortex_GD.py
```

To run it in the background (**Strongly recommended to run all the program in the background.**):
```
nohup python -O GP_vortex_GD.py &
```

Important physical parameters in `GP_vortex_GD.py`:
```python
N = 8                     # Number of qubits in x and y directions
x1, x2 = -6, 6            # Spatial domain boundaries
Ndx = 2**N
dx = (x2 - x1) / Ndx      # Grid displacement

g = 100 / dx**2           # Nonlinear interaction parameters
omega = 0.8               # Rotation frequency
maxdim = 10               # Bond dimension for the MPS
maxdim_psi2 = 1000000000  # Bond dimension for nonlinear term
cutoff_mps = 1e-12        # wave function cutoff
cutoff_mps2 = 1e-8        # density function (abs(psi)^2) cutoff 
psi2_update_length = 1
steps = 10                # Number of optimization steps
step_iter = 5             # Interval to autosave output
dt = dx**2 / 2            # Time displacement (any positive value for GD)
```

After the simulation completes, run:
```
python GD_catch_data.py
```

Make sure `steps` and `step_iter` in `GD_catch_data.py` match those in `GP_vortex_GD.py`.

To visualize the density (i.e., `abs(psi)^2`):
```
python plot2D_sec.py
```

---

### 2. `IMT_TDVP/`: Imaginary Time Evolution (TDVP)

This folder implements imaginary time evolution for solving the 2D rotating GPE using the TDVP method.
Same physical parameters as in the GD solver.
Run:
```
python -O TDVP.py
```

After completion, visualize the result:
```
python TDVP_catch_data.py
```

However, for onsidering the algorithm stability, the time step `dt` must satisfy:
```python
dt <= dx**2 / 2
```

---

### 3. `REAL_TIME_TDVP/`: Real-Time Evolution

This folder implements real-time evolution of the 2D GPE using TDVP.

Run:
```
python -O RT_TDVP.py
```

Then plot observables:
```
python plot_bondim.py   # Bond dimension vs Steps
python plot_pos.py      # Expectation value of position vs time (steps * dt)
```

In `RT_TDVP.py`, you can specify a saved MPS as initial state. Here we use 
the `GD2_mps_step_3000_input.pkl` to be an example:
```python
psi = get_init_other_state(N, x1, x2, maxdim, "GD2_mps_step_3000_input.pkl")
```

Make `GD2_mps_step_3000_input.pkl` sure the file exists.

---

### 4. `NumpyTensorTools/`: Source Modules

This folder contains core tensor network algorithm modules for:

- Gradient Descent
- TDVP (Time-Dependent Variational Principle)
- DMRG (Density Matrix Renormalization Group)

Feel free to explore this folder if you want to understand the tensornetwork algorithm logic.

### 5. Acknowledge
The present implementation has been validated through 1,145,141,919,810 independent test cases.
We acknowledge Mr. Koji Tadokoro for his inspiration throughout the development process. Here is a picture of him when he was younger.
```                                                                                                                                                                                                                                                                                                       
      $&&  &&&  &&&                          +$x;:::; ::.                                           
       &&   &. &&&                    &&&&&&&&&&&&&&&&&&&&&&&;                                      
    &&&&&&&&&&&&&&&&&&            X&&&&&&&&&&&&&&&&&&&&&&&&&&&&X.                                   
    &&::xxxxxxxx$  &&&         ;&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                                
    :; +&&&&&&&&&& :;       .&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                           
           ;&&&$           X&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&           &&&&&$          
   :&&&&&&&&&&&&&&&&&&    X&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&+       &&&X.;&&&X        
            &&           &&&&&&&&&&&&&&&&&&&&&&&&&&&&X&&&&&&&&&&&&&&&&&&&&&      &&&     X&&        
         &&&&&          x&&&&&&&&&&&&&&&&&&&&xX&+;:....:..:X&&&&&&&&&&&&&&&              $&&        
         +&&&+          &&&&&&&&&&&&&&&&&+++;:.++:.......xXX+:::;X&&&&&&&&&&            &&&X        
            &&+        &&&&&&&&&&&&&&&&&++;........................:X&&&&&&&&        &&&&&          
      x&&   &&+         &&&&&&&&&&&&&&&++++:.........................;+X&&&&&:     &&&&             
      &&&&&&&&&&&&&&&  $&&&&&&&&&&&&&&$+++++..........................+++&&&&x   .&&&               
    +&&&$$$&&&&$$$$$$  &&&&&&&&&&&&&&&X+++++;.........................;++&&&&;   &&&&&&&&&&&X       
    &&+     &&+        X&&&&&&&&&&&&&&&X+++++:........................:++x&&&:                      
      X&&&&&&&&&&&&&    &&&&&&&&&&&&&&+++++++;;:.......................;+x&&&           ..          
       ;;;;+&&&;;;;;    &&&&&&&&&&&&x++++++++&&&&&&&&&&+..............:;+x&&$         &&&&&         
            &&+         &&&&&&&&&&&X++++++++;.....:;;;;+:....   .xX&&&&&&&X&         &&&&&&         
    &&&&&&&&&&&&&&&&&&  &$X&&&&&&&&xx+++++:..........::................:++x&       &&&  &&&         
    &&&&&&&&&&&&&&&&&&  &xx&+.&&&&&xx+++++;.....:X&&&:.&;......+:.......;+x+      &&X   &&&         
                       .X+++xXx:&&&xx+++++...+;&&&&   &:......&..$&X&;:$;;x:    X&&     &&&         
            .+&&&&&&+  :X+  ;++.&&xxx++++;.......::......;....:..;$&$ ;x::x     &&&&&&&&&&&&&       
    &&&&&&&&&&&&&&+    ;++&$x+; &&xxx++++ ...............;;...............:.            &&&         
            &&&    X&   ;X&xx+;.&$xxx++++:...............;+:...;:..........:            &&&         
          x&&.   $& .    ;Xxx+;:&xxxxx++++: ...........:++++....+..........X                        
         +&&:             $+xxx.xxxxxx+++++: .........&x+++++.  .x........:+          &&&           
         &&&               ;+;.;xxxxxx++++++;  ......X&x+X&&x;.;x.:.......+:     &&X  &&&&&&&&      
         &&&$               +x+x&Xxxxx+++++++;  .....;+++:.:+x+:xx.. .....+   &&&&&&&&&&&&&&&&&&    
          ;&&&&&&;            &xX&xxxx++++++++ .............++:..........:&             X&& $&x     
              $&&             &Xx&xxxx++++++++:.............;:...........X     &&&&&&&&&&&&&&&&&    
                               &x&Xxxx++++++++;...:..:;xX$$$XXXX+:......;      &&&::::::;&&.x&      
            &&&                ;xX&xxx+++++++++...+&+xxxx+;;;;xxXX&&:...$     :&&&&&&&&&&&&&&&      
    X&&&&&&&&&&&&&&&&          .+x&&xx+++++++++........:;xXx++x+:......X      X&& &&&&&&& &&&:      
    x&&$$$$$&&&++++xX           X+x&&xx++++++++..........:;;+;........;      .&&&+& &&&+&&&&&;&&    
        :&&&&&&                 &+++&&xx+++++++;.....................:+       &&  &&&&&.&&X$&&&&    
        &&+ x&&X                $++++x&Xx++++++++:..................:+                       $      
        &&: ;&&&                +X++++++&$++++++++++:..............:.                       &&&     
        ;&&&&&&;                +++++++++x&&++++++++++;.;++++++;..x                          &&&;   
           x&&&                 &;++++++++++X&&&+++++++++++++++++&                                  
          &&&                   x+++++++++++++xx$&&&x+++++++xX&&X+                                  
                               $++++++++++++++++xxx$$$XXx+++++++++                                  
                  &$&&       :&+++++++++++++++++++;;:.........:;;+                                  
                 $x  &;   $&&&+++++++++++++++;;:................:+$                                 
                  X&&. $&&&&&&&&&&&&&&x;;:.......................:+&$                               
                    +&&&&&&&&&&&&&&&&&&&&&&&x;....................:x&&&X                            
              +&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$;:................:+&&$$&.                         
    .x$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$+.............:x&$$$$$&&&&&$X+.                
 X&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$&&$xx++xX$&&&$$$$$$$&$$$$$$$$$$$$$$+         
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$&&&&&&&&&$$$$$$$$$$$$$$&&&$$$$$$$$$$$$$$$$$$$$+     
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$$$X 
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
```                                                                                                  
                                                                                                                                                                                                                                            
     


