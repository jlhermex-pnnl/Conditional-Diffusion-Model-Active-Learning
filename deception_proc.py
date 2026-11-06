#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:49:51 2025

@author: john775
"""
import numpy as np
import os
import time


os.chdir('/people/hern856/GIFT_Paralell')

def compile_data(infile):
    
    pft_labels = ('W1_1','W1_2','W1_3','W2_1','W2_2','W2_3','W3_1','W3_2','W3_3',\
                  'W4_1','W4_2','W4_3','W5_1','W5_2','W5_3','W6_1','W6_2','W6_3',\
                  'W7_1','W7_2','W7_3','W8_1','W8_2','W8_3')

    pf_parameters = ('Electrical Conductivity','Total Tracer')
    
    
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
            if(len(line.split())>1):
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
                            print('found ',targ_lab,' ',par,' in ',fil)
                            obsdat[p,l,:] = simdat[:,col]
                
    return ert_times,ert_dat,obstimes,obsdat

while(True):
    
    #check to see if there is a list of running simulations
    done = True
    if (os.path.exists('./run_files.txt')):
        time.sleep(5)
        f1 = open('run_files.txt','r')
        run_files = f1.readlines()
        f1.close()
        os.system('rm run_files.txt')
        for fil in run_files:
            print(fil.rstrip())
            os.system('sbatch '+fil.rstrip())
        done = False
        rf_copy = run_files.copy()
        o_sizes = np.zeros(len(run_files))
        time.sleep(30)
    
        o_times = -np.ones(len(run_files))
        
    while(not done):
        for i,line in enumerate(run_files):
            pre = line.split('.')[0]
            if(pre!='done'):
                print(f'Checking run {pre}')
            if(os.path.exists(pre+'.out')):
                if(o_times[i]==-1):
                    o_times[i] = time.perf_counter()
                    o_sizes[i] = os.path.getsize(f'./{pre}.out')
                print(f'Found {pre}.out')
                f1 = open(pre+'.out','r')
                out = f1.readlines()
                f1.close()
                nl = len(out)
                last_line = out[nl-1].split()
                if(len(last_line) > 2):
                    if ((last_line[0]=='Wall') & (last_line[1]=='Clock')): 
                        print(f'Run {pre} complete. Gathering sim results.')
                        print(pre+'.in')
                        ertt,ertd,obst,obsd = compile_data(pre+'.in')
                        np.save(pre+'_ertt',ertt)
                        np.save(pre+'_ertd',ertd)
                        np.save(pre+'_obst',obst)
                        np.save(pre+'_obsd',obsd)
                        os.system(f'rm {pre}*.pft')
                        os.system(f'rm {pre}*.srv')
                        os.system(f'tar -cvf {pre}.tar {pre}*.npy')
                        os.system(f'rm {pre}*.npy')
                        run_files[i] = 'done.txt'
            
                #check pflotran output file to make sure it's growing
                #else respawn the job
                fsize = os.path.getsize(f'./{pre}.out')
                c_time = time.perf_counter()
                print('\to_time: ',str(o_times[i]))
                print('\ttime_delta(sec): ',str(c_time-o_times[i]))
                print('\tfile_sizes: ',str(o_sizes[i]),str(fsize))
                if ((o_times[i]!=-1) and (fsize == o_sizes[i]) and ((c_time-o_times[i])>120)):
                    print(f'Re_spawning {rf_copy[i]}')
                    os.system(f'rm {pre}.out {pre}*.pft {pre}*.dat {pre}*.srv')
                    os.system('sbatch '+rf_copy[i].rstrip())
                    o_sizes[i] = 0
                else:
                    if(fsize>o_sizes[i]):
                        o_times[i] = c_time
                    o_sizes[i] = fsize
                    
                    
        done = True
        for line in run_files:
            if(line != 'done.txt'):
                done = False
                
        if(done):
            os.system('scancel -u hern856')   #cancels any jobs that are hung up
        else:
            print(f"Waiting for 30 seconds for more runs to complete")
            time.sleep(30)
                       
        
    
    if done:
        print(f"Done processing: Waiting 10 seconds for new run file list.")
        (time.sleep(10))
        
        
    
    
    