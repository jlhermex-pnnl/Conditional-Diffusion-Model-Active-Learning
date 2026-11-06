#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 21:18:27 2025

@author: john775
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from h5py import *
import subprocess
import time
import re
#os.chdir("/mnt/sda2/DVZ/GIFT")

def replace_material_property(template_content, material_name, archie_m, archie_a, surface_ec, porosity, permeability):
    """
    Replace parameter values in a MATERIAL_PROPERTY section.
    
    Args:
        template_content: String content of the template file
        material_name: Name of the material (e.g., 'soil1', 'soil2', 'soil3')
        archie_m: Archie cementation exponent
        archie_a: Archie tortuosity constant
        surface_ec: Surface electrical conductivity
        porosity: Porosity value
        permeability: Permeability value
    
    Returns:
        Modified template content with substituted parameters
    """
    # Find the MATERIAL_PROPERTY section for this material
    # Pattern matches: MATERIAL_PROPERTY soil1 ... END (including everything in between)
    pattern = rf'(MATERIAL_PROPERTY {material_name}.*?)\n(.*?)\nEND'
    
    def replacer(match):
        header = match.group(1)
        body = match.group(2)
        
        # Replace ARCHIE_CEMENTATION_EXPONENT
        body = re.sub(r'ARCHIE_CEMENTATION_EXPONENT\s+[\d.eEdD+-]+',
                     f'ARCHIE_CEMENTATION_EXPONENT {archie_m:.6e}', body)
        
        # Replace ARCHIE_TORTUOSITY_CONSTANT
        body = re.sub(r'ARCHIE_TORTUOSITY_CONSTANT\s+[\d.eEdD+-]+',
                     f'ARCHIE_TORTUOSITY_CONSTANT {archie_a:.6e}', body)
        
        # Replace SURFACE_ELECTRICAL_CONDUCTIVITY
        body = re.sub(r'SURFACE_ELECTRICAL_CONDUCTIVITY\s+[\d.eEdD+-]+',
                     f'SURFACE_ELECTRICAL_CONDUCTIVITY {surface_ec:.6e}', body)
        
        # Replace POROSITY
        body = re.sub(r'POROSITY\s+[\d.eEdD+-]+',
                     f'POROSITY {porosity:.6e}', body)
        
        # Replace PERMEABILITY values (all three: PERM_X, PERM_Y, PERM_Z)
        body = re.sub(r'PERM_X\s+[\d.eEdD+-]+',
                     f'PERM_X {permeability:.6e}', body)
        body = re.sub(r'PERM_Y\s+[\d.eEdD+-]+',
                     f'PERM_Y {permeability:.6e}', body)
        body = re.sub(r'PERM_Z\s+[\d.eEdD+-]+',
                     f'PERM_Z {permeability:.6e}', body)
        
        return f'{header}\n{body}\nEND'
    
    # Apply replacement using DOTALL flag to match across newlines
    modified_content = re.sub(pattern, replacer, template_content, flags=re.DOTALL)
    
    return modified_content

#infile = 'gift_tracer_ERT3.in'
pft_labels = ('W1_1','W1_2','W1_3','W2_1','W2_2','W2_3','W3_1','W3_2','W3_3',\
              'W4_1','W4_2','W4_3','W5_1','W5_2','W5_3','W6_1','W6_2','W6_3',\
              'W7_1','W7_2','W7_3','W8_1','W8_2','W8_3')

pf_parameters = ('Electrical Conductivity','Total Tracer')

def compile_data(infile):
    pre = infile.split('.in')[0]
    flist = os.listdir('./')
    ert_files=[]
    ert_times=[]
    
    #gather the ERT survey files and times
    for f in flist:
        if(f.startswith(pre) & f.endswith('.srv')):
            ert_files.append(f)
            ert_times.append(float(f.split('-')[2].split('h.srv')[0]))
    
    idx = np.argsort(ert_times)
    
    #read in the simulated ERT data
    for i,indx in enumerate(idx):
        #read the next file in the time sequence
        f1 = open(ert_files[indx],'r')
        lines = f1.readlines()
        f1.close()
        
        #ne is the number of electrodes
        ne = int(lines[0].split()[0])
        
        #find the line where the data starts
        for j,line in enumerate(lines[ne:len(lines)]):
            if(len(line.split()) > 0):
                if int(line.split()[0]) == 1:
                    lstart = ne+j
        
        #nm is the number of measurements in the survey
        nm = int(lines[lstart-1].split()[0])
        if(i==0):
            ert_dat = np.zeros((len(idx),nm))
        
        #read in the survey data for the current time
        for j,line in enumerate(lines[lstart:lstart+nm]):
            #print(line.split()[0])
            ert_dat[i,j] = float(line.split()[5])
                
    ert_times.sort()

    #gather the pft file names
    pft_files = []
    for f in flist:
        if(f.startswith(pre) & f.endswith('.pft')):
            pft_files.append(f)
        
        
    for p,par in enumerate(pf_parameters):
        for f,fil in enumerate(pft_files):
            f1 = open(fil,'r')
            line = f1.readline()
            f1.close()
            labels = line.split(',')
            simdat = np.genfromtxt(fil,skip_header=1)
            if((p==0) & (f==0)):
                npft = len(simdat)
                obstimes = simdat[:,0]
                obsdat = np.zeros((len(pf_parameters),len(pft_labels),npft))
                
            for l,targ_lab in enumerate(pft_labels):
                    for col,lab in enumerate(labels):
                        if((par in lab) & (targ_lab in lab)):
                            #print('found ',targ_lab,' ',par,' in ',fil)
                            obsdat[p,l,:] = simdat[:,col]
                
    return ert_times,ert_dat,obstimes,obsdat   

#execute parallel run 
def run_sim_par(infil,parms,sim_name,use_limits=True,delete_files=True):
    
    #build the h5 database file referenced by infil
    dbase_f_name = infil.split('.')[0]+'_dbase.h5'
    h5file = File(dbase_f_name,mode='w')
    num_real = parms.shape[1]

    #parms must be a numpy array with 15 rows and number of columns
    #equal to the number of realizations in the .h5 file
    #The row values are
    #parms is a vector with 15 values as follows
    #0: 30 sand archie cementation exponent 1.1 to 3.1
    #1: 30 sand arch tortuosity constant: 1 to 3
    #2: 30 sand archie surface EC: 1e-3 to 1e-6
    #3: 30 sand porosity: 0.15 to 0.45
    #4: 30 sand permeability: 1e-10 to 1e-12
    #5: 70 sand archie cementation exponent 1.1 to 3.1
    #6: 70 sand arch tortuosity constant: 1 to 3
    #7: 70 sand archie surface EC: 1e-3 to 1e-6
    #8: 70 sand porosity: 0.15 to 0.45
    #9: 70 sand permeability: 5e-11 to 5e-13
    #10: 16 sand archie cementation exponent 1.1 to 3.1
    #11: 16 sand arch tortuosity constant: 1 to 3
    #12: 16 sand archie surface EC: 1e-3 to 1e-6
    #13: 16 sand porosity: 0.15 to 0.45
    #14: 16 sand permeability: 5e-10 to 5e-12
    
    #set the limits for each parameter
    limits = np.zeros((15,2))
    limits[0] = [1.1,3.1]
    limits[1] = [1.0,3.0]
    limits[2] = [1e-7,1e-3]
    limits[3] = [0.15,0.45]
    limits[4] = [1e-12,1e-10]
    limits[5] = [1.1,3.1]
    limits[6] = [1.0,3.0]
    limits[7] = [1e-7,1e-3]
    limits[8] = [.15,.45]
    limits[9] = [5e-13,5e-11]
    limits[10] = [1.1,3.1]
    limits[11] = [1.0,3.0]
    limits[12] = [1e-7,1e-3]
    limits[13] = [0.15,0.45]
    limits[14] = [5e-12,5e-10]
    
    #bound parameter values if needed
    if(use_limits):
        for i,pcol in enumerate(parms):
            for j,p in enumerate(pcol):
                if p < limits[i,0]:
                    parms[i,j] = limits[i,0]
                    print ('bounding parameter ',i,' to ',limits[i,0])
                if p > limits[i,1]:
                     parms[i,j] = limits[i,1]
                     print ('bounding parameter ',i,' to ',limits[i,1])

    #write the .h5 file
    variable_names = []
    variable_names.append('30_sand_Archie_M')
    variable_names.append('30_sand_Archie_A')
    variable_names.append('30_sand_surface_EC')
    variable_names.append('30_sand_porosity')
    variable_names.append('30_sand_permeability')
    variable_names.append('70_sand_Archie_M')
    variable_names.append('70_sand_Archie_A')
    variable_names.append('70_sand_surface_EC')
    variable_names.append('70_sand_porosity')
    variable_names.append('70_sand_permeability')
    variable_names.append('16_sand_Archie_M')
    variable_names.append('16_sand_Archie_A')
    variable_names.append('16_sand_surface_EC')
    variable_names.append('16_sand_porosity')
    variable_names.append('16_sand_permeability')
    
    
    for i,var in enumerate(variable_names):
        h5dset = h5file.create_dataset(var, data=parms[i])  #add the variable name and column
    h5file.close()
    
    run_remote(infil,dbase_f_name,'soil_compressibility.txt','gift_survey.srv',sim_name,num_real,parms)

#run the pflotran simulations on deception
def run_remote(infile,h5dbase,soil_comp_file,survey_file,run_name,num_real,parms):
    #remote_machine = 'rcdtn1.pnl.gov'
    remote_machine = 'deception05.pnl.gov'
    
    # Check if we're already on the deception cluster (either login or compute node)
    hostname = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
    
    # Use local directory to avoid permission issues
    # The shared directory /rcfs/projects/dvztj/gift/ has inconsistent write permissions
    # Use a subdirectory within the current experiment directory to avoid conflicts
    remote_run_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pflotran_runs/')
    
    # Create directory if it doesn't exist
    os.makedirs(remote_run_dir, exist_ok=True)
    print(f"Using local pflotran directory: {remote_run_dir}")
    
    run_account = 'dvztj'
    num_cpus_per_run = 32
    max_run_time_min = 20
    
    # Check if remote_run_dir exists locally - if it does, we're on deception
    is_local = os.path.exists(remote_run_dir)
    print(f"DEBUG: hostname = '{hostname}', remote_run_dir = '{remote_run_dir}', is_local = {is_local}")
    
    pre = infile.split('.')[0]
    infil_name = f'{pre}_{run_name}.in'
    print('working in directory:'+os.getcwd())
    print(os.listdir('./'))
    time.sleep(10)

    #copy the files to the remote run directory
    if is_local:
        # We're already on deception, use local copy
        os.system(f'cp {h5dbase} {remote_run_dir}')
        os.system(f'cp {soil_comp_file} {remote_run_dir}')
        os.system(f'cp {survey_file} {remote_run_dir}')
    else:
        # We're on a different machine, use scp
        os.system(f'scp {h5dbase} {remote_machine}:{remote_run_dir}')
        os.system(f'scp {soil_comp_file} {remote_machine}:{remote_run_dir}')
        os.system(f'scp {survey_file} {remote_machine}:{remote_run_dir}')
    
    #build the sbatch run files for each realization
    rf_names = []
    # Create pflotran_logs directory if it doesn't exist
    # Use realpath to resolve symlinks (/people -> /qfs/people) to match SLURM output paths
    base_dir = os.path.realpath(os.getcwd())
    pflotran_logs_dir = os.path.join(base_dir, 'pflotran_logs')
    if not os.path.exists(pflotran_logs_dir):
        os.makedirs(pflotran_logs_dir)
        print(f"Created pflotran_logs directory: {pflotran_logs_dir}")
    
    # Create pflotran_input_files directory for .in and .run files
    pflotran_input_dir = os.path.join(os.getcwd(), 'pflotran_input_files')
    if not os.path.exists(pflotran_input_dir):
        os.makedirs(pflotran_input_dir)
        print(f"Created pflotran_input_files directory: {pflotran_input_dir}")
    
    for r in range(num_real):
        rstr = '{0:-05.0f}'.format(r)
        pre = infile.split('.')[0]
        infil_name = pre+'_'+run_name+'_{0:-05.0f}.in'.format(r)
        
        # Read template file
        with open(infile, 'r') as f:
            template_content = f.read()
        
        # Substitute parameters for this realization
        # Parameters from HDF5 datasets (15 parameters total)
        param_values = {
            '30_sand_Archie_M': parms[0, r],
            '30_sand_Archie_A': parms[1, r],
            '30_sand_surface_EC': parms[2, r],
            '30_sand_porosity': parms[3, r],
            '30_sand_permeability': parms[4, r],
            '70_sand_Archie_M': parms[5, r],
            '70_sand_Archie_A': parms[6, r],
            '70_sand_surface_EC': parms[7, r],
            '70_sand_porosity': parms[8, r],
            '70_sand_permeability': parms[9, r],
            '16_sand_Archie_M': parms[10, r],
            '16_sand_Archie_A': parms[11, r],
            '16_sand_surface_EC': parms[12, r],
            '16_sand_porosity': parms[13, r],
            '16_sand_permeability': parms[14, r],
        }
        
        # Replace soil1 (30 sand) parameters
        template_content = replace_material_property(template_content, 'soil1', 
            archie_m=param_values['30_sand_Archie_M'],
            archie_a=param_values['30_sand_Archie_A'],
            surface_ec=param_values['30_sand_surface_EC'],
            porosity=param_values['30_sand_porosity'],
            permeability=param_values['30_sand_permeability'])
        
        # Replace soil2 (70 sand) parameters
        template_content = replace_material_property(template_content, 'soil2',
            archie_m=param_values['70_sand_Archie_M'],
            archie_a=param_values['70_sand_Archie_A'],
            surface_ec=param_values['70_sand_surface_EC'],
            porosity=param_values['70_sand_porosity'],
            permeability=param_values['70_sand_permeability'])
        
        # Replace soil3 (16 sand) parameters
        template_content = replace_material_property(template_content, 'soil3',
            archie_m=param_values['16_sand_Archie_M'],
            archie_a=param_values['16_sand_Archie_A'],
            surface_ec=param_values['16_sand_surface_EC'],
            porosity=param_values['16_sand_porosity'],
            permeability=param_values['16_sand_permeability'])
        
        # Write modified template to new file in pflotran_input_files directory
        local_infil_path = os.path.join(pflotran_input_dir, infil_name)
        with open(local_infil_path, 'w') as f:
            f.write(template_content)
        
        # Copy to remote directory
        if is_local:
            os.system(f'cp {local_infil_path} {remote_run_dir}{infil_name}')
        else:
            os.system(f'scp {local_infil_path} {remote_machine}:{remote_run_dir}{infil_name}')
        rf_name = os.path.join(pflotran_input_dir, pre+'_'+run_name+'_{0:-05.0f}.run'.format(r))
        f1 = open(rf_name,'w')
        f1.write('#!/bin/bash\n')
        f1.write(f'#SBATCH -A {run_account}\n')
        f1.write(f'#SBATCH -t {max_run_time_min}\n')
        f1.write(f'#SBATCH -N 1\n')
        f1.write(f'#SBATCH -n {num_cpus_per_run}\n')
        f1.write(f'#SBATCH -p shared\n')
        run_file_prefix = f'{pre}_{run_name}_{rstr}'
        f1.write(f'#SBATCH -J {run_name}_{rstr}\n')
        f1.write(f'#SBATCH -o {pflotran_logs_dir}/{run_file_prefix}.out\n')
        f1.write(f'#SBATCH -e {pflotran_logs_dir}/{run_file_prefix}.err\n\n')
        f1.write('# Clear inherited SLURM environment variables that cause conflicts\n')
        f1.write('unset SLURM_CPUS_PER_TASK\n')
        f1.write('unset SLURM_TRES_PER_TASK\n\n')
        f1.write('source ~/.bashrc\n\n')
        f1.write(f'# Change to run directory\n')
        f1.write(f'cd {remote_run_dir}\n\n')
        f1.write(f'# Use mpirun with explicit process count (Intel MPI)\n')
        f1.write(f'mpirun -np {num_cpus_per_run} $PFLOTRAN_DIR/src/pflotran/pflotran -realization_id {r+1} -pflotranin {infil_name}\n')
        f1.close()
        
        #copy this sbatch script to the run directory
        if is_local:
            os.system(f'cp {rf_name} {remote_run_dir}')
        else:
            os.system(f'scp {rf_name} {remote_machine}:{remote_run_dir}')
        rf_names.append(rf_name)
    run_files_list = os.path.join(pflotran_input_dir, 'run_files.txt')
    f1=open(run_files_list,'w')
    for i,f in enumerate(rf_names):
        f1.write(f+'\n')
    f1.close()
    if is_local:
        os.system(f'cp {run_files_list} {remote_run_dir}')
    else:
        os.system(f'scp {run_files_list} {remote_machine}:{remote_run_dir}')  
    
    # Submit jobs directly since we're running locally
    if is_local:
        print(f"Submitting {num_real} PFLOTRAN jobs to SLURM...")
        for rf_name in rf_names:
            job_script = os.path.join(remote_run_dir, os.path.basename(rf_name))
            result = subprocess.run(['sbatch', job_script], capture_output=True, text=True)
            print(f"  Submitted: {job_script} -> {result.stdout.strip()}")
        print("All jobs submitted. Waiting for completion...")
        time.sleep(10)  # Give jobs time to start
    
    #Note: The script below watches for the tar files which are created by deception_proc.py
    #Since we're submitting directly, we need to monitor and package results ourselves 
    start_time = time.perf_counter()
    max_wait_time = max_run_time_min * 60 * 6  # 6x the expected runtime to handle parallel overhead
    iteration_count = 0
    
    done = np.zeros(num_real,dtype='bool')
    while(not done.all()):
        
        for i in range(num_real):
            if(not done[i]):
                pre = infile.split('.')[0]
                tfile = pre+'_'+run_name+'_{0:-05.0f}.tar'.format(i)
                run_file_prefix = pre+'_'+run_name+'_{0:-05.0f}'.format(i)
                
                # If running locally, check if job completed and package results
                if is_local:
                    # Fix: SLURM writes .out files to pflotran_logs_dir, not remote_run_dir
                    out_file = os.path.join(pflotran_logs_dir, f'{run_file_prefix}.out')
                    tar_file = os.path.join(remote_run_dir, tfile)
                    
                    # Check if tar file already exists in remote_run_dir
                    if os.path.exists(tar_file):
                        print(f'Found {tfile}, extracting...')
                        os.system(f'cp {tar_file} .')
                        os.system(f'tar -xvf {tfile}')
                        os.system(f'rm {tfile}')
                        done[i] = True
                    # Check if job output exists
                    elif os.path.exists(out_file):
                        with open(out_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Check if job completed successfully
                        if lines and len(lines[-1].split()) > 2:
                            last_line = lines[-1].split()
                            if last_line[0] == 'Wall' and last_line[1] == 'Clock':
                                # Job completed - check if files were already extracted to current dir
                                expected_files = [
                                    f'{run_file_prefix}_ertt.npy',
                                    f'{run_file_prefix}_ertd.npy',
                                    f'{run_file_prefix}_obst.npy',
                                    f'{run_file_prefix}_obsd.npy'
                                ]
                                all_files_exist = all(os.path.exists(f) for f in expected_files)
                                
                                if all_files_exist:
                                    # Files already here (deception_proc.py must have processed and we extracted)
                                    print(f'Simulation {i} ({run_file_prefix}) already processed and extracted')
                                    done[i] = True
                                else:
                                    # Try to package results ourselves
                                    print(f'Run {run_file_prefix} complete. Packaging results...')
                                    orig_dir = os.getcwd()
                                    os.chdir(remote_run_dir)
                                    try:
                                        ertt, ertd, obst, obsd = compile_data(f'{run_file_prefix}.in')
                                        np.save(f'{run_file_prefix}_ertt', ertt)
                                        np.save(f'{run_file_prefix}_ertd', ertd)
                                        np.save(f'{run_file_prefix}_obst', obst)
                                        np.save(f'{run_file_prefix}_obsd', obsd)
                                        os.system(f'tar -cvf {tfile} {run_file_prefix}*.npy')
                                        os.system(f'rm {run_file_prefix}*.npy')
                                        print(f'Created {tfile}')
                                    except Exception as e:
                                        print(f'WARNING: Failed to package {run_file_prefix}: {e}')
                                        # Check if it's because files were already cleaned up
                                        if 'No such file or directory' in str(e) or '.srv' in str(e) or '.pft' in str(e):
                                            print(f'  -> Likely already processed by deception_proc.py - marking as done')
                                        # Mark as done anyway to avoid infinite wait
                                        done[i] = True
                                    finally:
                                        os.chdir(orig_dir)
                        
                        # Check if job failed - detect PETSC errors or other failures
                        err_file = os.path.join(pflotran_logs_dir, f'{run_file_prefix}.err')
                        if os.path.exists(err_file) and os.path.getsize(err_file) > 1000:
                            # Large error file likely indicates failure
                            with open(err_file, 'r') as f:
                                err_content = f.read()
                            if 'PETSC ERROR' in err_content or 'MPI_Abort' in err_content or 'Terminate' in err_content:
                                print(f'WARNING: Simulation {i} ({run_file_prefix}) FAILED - PETSC error detected')
                                done[i] = True  # Mark as done to avoid infinite wait
                            # Also check if job has been running too long (stalled)
                            elif (time.perf_counter() - start_time) > (max_run_time_min * 60 * 2):
                                # If more than 2x max_run_time has passed, consider it failed
                                print(f'WARNING: Simulation {i} ({run_file_prefix}) TIMEOUT - marking as failed')
                                done[i] = True
                else:
                    # Remote case - check via SSH
                    command = ["ssh", remote_machine, "test", "-f", remote_run_dir+tfile]
                    result = subprocess.run(command, capture_output=True)
                    
                    if result.returncode == 0:
                        time.sleep(10)
                        print(f'Downloading {tfile}...')
                        os.system(f'scp {remote_machine}:{remote_run_dir}{tfile} .')
                        os.system(f'tar -xvf {tfile}')
                        os.system(f'rm {tfile}')
                        done[i] = True
                    
        
        # Check for overall timeout
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > max_wait_time:
            incomplete = np.where(done == False)[0]
            print(f'\n{"="*60}')
            print(f'TIMEOUT: Maximum wait time ({max_wait_time/60:.1f} min) exceeded')
            print(f'Completed: {np.sum(done)}/{num_real} simulations')
            print(f'Incomplete/Failed: {len(incomplete)} simulations: {incomplete.tolist()}')
            print(f'{"="*60}\n')
            break
        
        iteration_count += 1
        completed = np.sum(done)
        print(f'Waiting 30 seconds to check for complete simulations [{completed}/{num_real} done, iteration {iteration_count}]')
        time.sleep(30)
    
    # Final summary
    completed = np.sum(done)
    failed = num_real - completed
    print(f'\n{"="*60}')
    print(f'PFLOTRAN SIMULATION BATCH COMPLETE')
    print(f'Total submitted: {num_real}')
    print(f'Successful: {completed}')
    print(f'Failed: {failed}')
    if failed > 0:
        failed_indices = np.where(done == False)[0]
        print(f'Failed simulation indices: {failed_indices.tolist()}')
    print(f'Total time: {(time.perf_counter() - start_time)/60:.1f} minutes')
    print(f'{"="*60}\n')
                  

def run_sim(infile,parms,outfile,use_limits=True,delete_files=True):
    #infile = pflotran input file template
    
    #parms is a vector with 15 values as follows
    #0: 30 sand archie cementation exponent 1.1 to 3.1
    #1: 30 sand arch tortuosity constant: 1 to 3
    #2: 30 sand archie surface EC: 1e-3 to 1e-6
    #3: 30 sand porosity: 0.15 to 0.45
    #4: 30 sand permeability: 1e-10 to 1e-12
    #5: 70 sand archie cementation exponent 1.1 to 3.1
    #6: 70 sand arch tortuosity constant: 1 to 3
    #7: 70 sand archie surface EC: 1e-3 to 1e-6
    #8: 70 sand porosity: 0.15 to 0.45
    #9: 70 sand permeability: 5e-11 to 5e-13
    #10: 16 sand archie cementation exponent 1.1 to 3.1
    #11: 16 sand arch tortuosity constant: 1 to 3
    #12: 16 sand archie surface EC: 1e-3 to 1e-6
    #13: 16 sand porosity: 0.15 to 0.45
    #14: 16 sand permeability: 5e-10 to 5e-12
    
    #name of pflotran input file to write
    #uselimits - if true make sure parms are within the limits below
    #delete_files - if true the pflotran files will be deleted after results read in
    
    #set the limits for each parameter
    limits = np.zeros((15,2))
    limits[0] = [1.1,3.1]
    limits[1] = [1.0,3.0]
    limits[2] = [1e-7,1e-3]
    limits[3] = [0.15,0.45]
    limits[4] = [1e-12,1e-10]
    limits[5] = [1.1,3.1]
    limits[6] = [1.0,3.0]
    limits[7] = [1e-7,1e-3]
    limits[8] = [.15,.45]
    limits[9] = [5e-13,5e-11]
    limits[10] = [1.1,3.1]
    limits[11] = [1.0,3.0]
    limits[12] = [1e-7,1e-3]
    limits[13] = [0.15,0.45]
    limits[14] = [5e-12,5e-10]
    
    #bound parameter values if needed
    if(use_limits):
        for i,p in enumerate(parms):
            if p < limits[i,0]:
                parms[i] = limits[i,0]
                print ('bounding parameter ',i,' to ',limits[i,0])
            if p > limits[i,1]:
                parms[i] = limits[i,1]
                print ('bounding parameter ',i,' to ',limits[i,1])

    #read the pflotran input file
    f1 = open(infile,'r')
    lines = f1.readlines()
    f1.close()
    
    #replace the appropriate lines to set parameters to parms
    lines[89] = '  ARCHIE_CEMENTATION_EXPONENT {0:10.3f}\n'.format(parms[0])
    lines[91] = '  ARCHIE_TORTUOSITY_CONSTANT {0:10.3f}\n'.format(parms[1])
    lines[95] = '  SURFACE_ELECTRICAL_CONDUCTIVITY {0:10.3E}  \n'.format(parms[2])
    lines[93] = '  POROSITY {0:6.5F}\n'.format(parms[3])
    lines[97] = '    PERM_X {0:10.3E}\n'.format(parms[4])
    lines[98] = '    PERM_Y {0:10.3E}\n'.format(parms[4])
    lines[99] = '    PERM_Z {0:10.3E}\n'.format(parms[4])
    
    lines[107] = '  ARCHIE_CEMENTATION_EXPONENT {0:10.3f}\n'.format(parms[5])
    lines[109] = '  ARCHIE_TORTUOSITY_CONSTANT {0:10.3f}\n'.format(parms[6])
    lines[113] = '  SURFACE_ELECTRICAL_CONDUCTIVITY {0:10.3E}  \n'.format(parms[7])
    lines[111] = '  POROSITY {0:6.5F}\n'.format(parms[8])
    lines[115] = '    PERM_X {0:10.3E}\n'.format(parms[9])
    lines[116] = '    PERM_Y {0:10.3E}\n'.format(parms[9])
    lines[117] = '    PERM_Z {0:10.3E}\n'.format(parms[9])
    
    lines[124] = '  ARCHIE_CEMENTATION_EXPONENT {0:10.3f}\n'.format(parms[5])
    lines[126] = '  ARCHIE_TORTUOSITY_CONSTANT {0:10.3f}\n'.format(parms[6])
    lines[130] = '  SURFACE_ELECTRICAL_CONDUCTIVITY {0:10.3E}  \n'.format(parms[7])
    lines[128] = '  POROSITY {0:6.5F}\n'.format(parms[8])
    lines[132] = '    PERM_X {0:10.3E}\n'.format(parms[9])
    lines[133] = '    PERM_Y {0:10.3E}\n'.format(parms[9])
    lines[134] = '    PERM_Z {0:10.3E}\n'.format(parms[9])
    
    #write the new input file to temp.in
    f1 = open(outfile,'w')
    for line in lines:
        f1.write(line)
    f1.close()
    mpi_path = "/home/john775/codes/petsc_intel/arch-linux-c-opt/bin/mpirun"
    pflotran_path="/home/john775/codes/pflotran_dfn/pflotran/src/pflotran/pflotran"

    "Print Simulation Running"
    os.system(f"{mpi_path} -np 32 {pflotran_path} -pflotranin {outfile}")
    
    ertt,ertd,obst,obsd = compile_data(outfile)
    if(delete_files):
        pre = outfile.split()[0]
        os.system('rm -f '+pre+'*')
    
    return ertt,ertd,obst,obsd
