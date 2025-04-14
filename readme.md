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

To run it in the background (**strongly recomend running in background.**):
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
                                                                                                                                                      
           X&   +&&&     &&&;                                                                                                                         
         :&&&&  :&&&&   &&&&&                                      X&&&&&&&&&&&&&&&&&&                                                                
          :&&&;  x&&$  &&&&&                              &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&x                                                           
      &&&&&&&&&&&&&&&&&&&&&&&&&&$                      &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&+                                                       
      &&&&&&&&&&&&&&&&&&&&&&&&&&$                 ;&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&+                                                     
      &&&&                   &&&&              ;&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&X                                               
      &&&& &&&&&&&&&&&&&&&;  &&&$             ;&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&x++                                            
           &&&&&&&&&&&&&&&&                &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                                         
                 ;&&&&&&.                .&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&X                  ;xxx:                 
     &&&&&&&&&&&&&&&&&&&&&&&&&&&&        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&x             &&&&&&&&&&&&              
     &&&&&&&&&&&&&&&&&&&&&&&&&&&&      x&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&           &&&&&$  x&&&&&:            
                 x&&&:                &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&         &&&&X      :&&&&            
                 X&&&:               &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&x+xx+;;;;;;;;;;&&&&&&&&&&&&&&&&&&&&&&&$         ;&&&&        &&&&X           
             &&&&&&&&:              x&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$;+&&;;;;.......... ..x&&&&&&&&&&&&&&&&&&&&&&&                     &&&&x           
              &&&&&&&               &&&&&&&&&&&&&&&&&&&&&&&&&&x;;;;;;:+&;.............x$&&&$+;;;X&&&&&&&&&&&&&&&&&                   &&&&&            
                                    &&&&&&&&&&&&&&&&&&&&&&&&&;;;;;......X;........................:;$&&&&&&&&&&&&&                 &&&&&&             
                  &&&$             &&&&&&&&&&&&&&&&&&&&&&&&$;;;;.....................................:+&&&&&&&&&&&&             &&&&&&&               
          &&&&    &&&$             &&&&&&&&&&&&&&&&&&&&&&&X;;;;;.......................................;X+&&&&&&&&&&         X&&&&&&:                 
         +&&&X    &&&$             &&&&&&&&&&&&&&&&&&&&&&$;;;;;;;.......................................;;;;&&&&&&&&        &&&&&&                    
        .&&&&&&&&&&&&&&&&&&&&&&    &&&&&&&&&&&&&&&&&&&&&&$;;;;;;;; ......................................;;;+$&&&&&&      ;&&&&$                      
       .&&&&&&&&&&&&&&&&&&&&&&&:   &&&&&&&&&&&&&&&&&&&&&&X;;;;;;;;: .....................................;;;;x&&&&&&      &&&&&&&&&&&&&&&&&           
      &&&&&       &&&&            &&&&&&&&&&&&&&&&&&&&&&$;;;;;;;;;;  .....................................;;;xX&&&&&     +&&&&&&&&&&&&&&&&&           
      &&&&        &&&$             &&&&&&&&&&&&&&&&&&&&&&X+x+;;;;;;;......................................;;;+x&&&&&                                  
       .&         &&&$             $&&&&&&&&&&&&&&&&&&&&&&+;;;;;;;;;.......................................;;;x&&&&&                                  
          &&&&&&&&&&&&&&&&&&&&     &&&&&&&&&&&&&&&&&&&&&x;;;;;;;;;;;;;;;...................................;;+x$&&&                &&&&&              
          &&&&&&&&&&&&&&&&&&&&     +&&&&&&&&&&&&&&&&&&x+;;;;;;;;;;;;x&&&&&&&&&&&$+.........................;;+xX&&&              :&&&&&&              
                  &&&$             X&&&&&&&&&&&&&&&&&xx;;;;;;;;;;;&&X.  :X&&&&&&&&&&+:.  ........ ..x&&&&&&&x+xx&&&             &&&&&&&&              
                  &&&$             X&&&&&&&&&&&&&&&&&xx;;;;;;;;;x:.............  ..:......      ..+X&&&&&&&&&&&X&&+           .&&&& &&&&              
                 .&&&&              &&&&&&&&&&&&&&&&&xxx;;;;;;;;...........................................;;;;x&&           &&&&+  &&&&              
     X&&&&&&&&&&&&&&&&&&&&&&&&&&&  :X&&&:.x&&&&&&&&&$xxx;;;;;;;:.................:x:........................;;;x$&          &&&&    &&&&              
     ;&&&&&&&&&&&&&&&&&&&&&&&&&&$  &&X;+x&&  $&&&&&&Xxxx;;;;;;;;........;&&&&&&&: .$;.........:;;............;;x$+        &&&&&     &&&&              
                                   x&x;;+xxX&  $&&&&xxxx;;;;;;;....+&X &&&&&&   x&...........X+...+&&&&&&&&;..;+&        &&&&       &&&&              
                             x+    x;&x;;;++++; &&&Xxxxx;;;;;;; ........ :X$&&&X$X......  ...& ..x&&&&&&   :&x;+$       &&&&        &&&&              
               .:+&&&&&&&&&&&&&    X+X.   .X;;. &&xxxxxx;;;;;;: ......................;..........  :+xx.+x;....+$       &&&&&&&&&&&&&&&&&&&&          
      &&&&&&&&&&&&&&&&&&&&&&&&&    X$;.Xx&&X;;  $&xxxxxx;;;;;;....................... ;;........................X       &&&&&&&&&&&&&&&&&&&&          
      x&&&&&&&&&&+ $&&&&&          &;;x&Xxx+;;. $&Xxxxxx;;;;;;........................;;......;.................+:                  &&&&              
                  &&&&$     +&&     +;&&xxxx;;. &&xxxxxx;;;;;;. ......................;;;..... X................;&                  &&&&              
                &&&&&    &&& .&&     +x&$xx;;;. &Xxxxxxx+;;;;;;  ...................:;;;;;..... X...............;X                  &&&&              
               &&&&.      &&          x;;;+&:;: &Xxxxxxxx;;;;;;;. ................;;&:;;;;:......X..............;x                                    
              &&&&:                    x;+xxxX..+xxxxxxxx+;;;;;;;.  .............X&;;;;;;;;:    .+:.............;X               &&&&                 
              &&&&                      :;;; ..:+Xxxxxxxx+;;;;;;;;;   ..........X&x+;;;;;;;;.......;............;&        &&&    &&&&&&&&&&&&;        
             .&&&&                      x.;;;.;;+Xxxxxxxx;;;;;;;;;;;     .   ..:X&$xxx&&&&&X+:+&&;.X.      ....;x         &&&    &&&&                 
              &&&&&                       $;;;;;&&xxxxxxx;;;;;;;;;;;;    .......;xxxx.....+XX+...X....   ......;&   +&&&&&&&&&&&&&&&&&&&&&&&&&&&      
               &&&&&&:                      &&&&X&xxxxxxx;;;;;;;;;;;;   ..................;:........... .......x    +&&&&&&&&&&&&&&&&&&&&&&&&&&&      
                 &&&&&&&&&                   &xxxXXxxxxxx;;;;;;;;;;;;. ...................;;;;......... ......:$                    .&&&X X&&&.       
                    X&&&&&                   &$xxX&xxxxxx;;;;;;;;;;;;: ...................+;:.................X       &&&&&&&&&&&&&&&&&&&&&&&&&&      
                                              &xxx&xxxxxx;;;;;;;;;;;;: ..............;xxxx.:x+; .............;x       &&&&X.........:&&&&..&          
                                              Xxxx$$xxxxx;;;;;;;;;;;;; ....+;..:+$&&&&&&&&&&&&&XXXx..........X        &&&&&&&&&&&&&&&&&&& X&&&        
                  &&&&                         &xxx&xxxxx+;;;;;;;;;;;;.....+&&xXXXXXXX+;;;;;;XXXXXX&&$......X         &&&&&&&&&&&&&&&&&&&&&&&X        
                  &&&&                         &xxxX&xxxx+;;;;;;;;;;;;............xXXXX;:::::XXXX+..:......;X         &&&& &&$&&&&&&& &&&&&&x         
      x&&&&&&&&&&&&&&&&&&&&&&&&$               +x+xx&&xxxx;;;;;;;;;;;;................;XXXXX;X$............&         X&&&& &&+&&&&$&&: &&&&$ &:       
      x&&&&&&&&&&&&&&&&&&&&&&&&x                $;;xx&Xxxx;;;;;;;;;;;;:...............:;;;;;;.............x         ;&&&&:&&& &&&& &&&&&&&&  &&&      
              +&&&&&&&                          &;;;+x&Xxxx;;;;;;;;;;;; .................................:X         &&&&& &&&&&&&& ;&&&&&&&&&&&&      
             &&&&&&&&&x                         +x;;;;x&&xxx;;;;;;;;;;;.................................;&           x&&   x &&&&&  &&&+ &&&&&&       
            &&&&  &&&&&.                         x;;;;;;X&xxx+;;;;;;;;;;;:.............................:&                                  +&:        
            &&&    &&&&&                         $;;;;;;;xX&xx+;;;;;;;;;;;;;..........................:X                                  &&&         
            &&&&  X&&&&&                         &;;;;;;;;;+X&Xx;;;;;;;;;;;;;;:......................+;                                   &&&&$       
            x&&&&&&&&&&+                         &;;;;;;;;;;;+X&&x;;;;;;;;;;;;;;;....;;;;;;;;;;;....X                                      $&&&&      
              :&&:&&&&&                          X;;;;;;;;;;;;;+xX&&+;;;;;;;;;;;;;;;;;;;;;;;;;;;;:.&                                         &&X      
                 &&&&&                          Xx;;;;;;;;;;;;;;;+xxX&&&+;;;;;;;;;;;;;;;;;;;;;;;;X&                                                   
              X&&&&&:                           &;;;;;;;;;;;;;;;;;;+xxxx$&&&X+;;;;;;;;;;;;;;;+X$&+&                                                   
                &&X                            &x;;;;;;;;;;;;;;;;;;;;+xxxxx&&&&&&&&X&&&&&&&$x+;;;;$                                                   
                                              &x;;;;;;;;;;;;;;;;;;;;;;;+xxxxxxXxxxx++++;;;;;;;;;;;x                                                   
                           &&&&&x            &X;;;;;;;;;;;;;;;;;;;;;;;;;;;+xxx+..............+;;;;++                                                  
                          &&   &&&       &&&&X;;;;;;;;;;;;;;;;;;;;;;;;;;;:......................:;;&                                                  
                          &&   .&&     &&&&&&;;;;;;;;;;;;;;;;;;;;;;;;;...........................;;X.                                                 
                          .&&&&&&   $&&&&&&&&&&&&&$Xxx+;;;;;;;;;:.................................;+&&                                                
                                  &&&&&&&&&&&&&&&&&&&&&&&&&$+. ....................................;x&&&&                                             
                              $&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&X;................................;x&&&&&&                                          
                        x&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&+ ............................;+&&XXX$&                                        
              .+&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&+..........................;;$$XXXXX&&$                                    
       ;$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$&X ....................:;x&XXXXXXXX$&&&&&&&&&X.                         
   &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&XXX$&X+.............+X&&$XXXXXXXXXXX$XXXXXXXXXXXXXXXXXX:                
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$XXXXX$&&&&&&&$$$XXXXXXXXXXXXXXX&XXXXXXXXXXXXXXXXXXXXXXXXX.           
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$XXXXXXX$$&&&&&&&&&&$XXXXXXXXXXXXXXXXXXXXXXXX&&&XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX        
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$XXXXXXXXXXXXXXXXXXXXX$&&&&&&$XXXXXXXXXXX$&&&&&$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx.  
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$XXXXXXXXXXXXXXXXXXXXXXXXXX$$$$$$$$$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx 
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx 
 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx 
                                                                                                                                                                                                                                            
     


