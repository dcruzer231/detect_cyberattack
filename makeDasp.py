"""
written by Daniel Cruz
This is the main program to convert
the datasets into DASP images.
"""


import numpy as np
import matplotlib.pyplot as plt
from dasp_functions import getHASP, getFASP
import h5py
import pandas as pd

"""
toy_virt_channels = {
    0 : 'index',
    1 : 'stepper_driver', //power consumption
    2 : 'ps_wall',       //power supply AC
    3 : 'driver_ps',    // driver to the power supply maps channel between power
    4 : 'instchan',     // pwm modulation from arduino to 
    5 : 'run_gt',       
    6 : 'inst_gt'       //flip-flop on starting on each instruction
}
"""

def showChannelDasp(run):
    instruction_im = np.zeros((256,256,3),dtype = "uint16")
    power_im = np.zeros((256,256,3),dtype = "uint16")
    
    #used to index the RGB channel for instruction and power images
    instindcount = 0
    pwrindcount = 0
    
    for i in range(1,5):
        plt.figure(i-1)
        daspArray = getHASP(run[i],T,True)
        if i == 1 or i == 4:
            instruction_im[:,:,instindcount] = (daspArray)
            instindcount+=1
        if i == 2 or i == 3:
            power_im[:,:,pwrindcount] = (daspArray)
            pwrindcount+=1
        plt.title(titles[i])
        plt.imshow(daspArray)
        # plt.imsave(savePath+goodFiles[0]+"_"+"1"+"_"+titles[i+1]+".png",daspArray)   
    
    #normalize to the 0-1 range 
    instruction_im = instruction_im/instruction_im.max()
    power_im = power_im/power_im.max()

    plt.figure(4)
    plt.title("instruction image")
    plt.imshow(instruction_im)
    # plt.imsave(savePath + goodFiles[0]+"_"+"1"+"_"+"instructionImage.png",instruction_im)

    plt.figure(5)
    plt.title("power image")
    plt.imshow(power_im)
    # plt.imsave(savePath + goodFiles[0]+"_"+"1"+"_"+"powerImage.png",power_im)

    hybrid = np.zeros_like(power_im)
    # average the stepper_driver and instchan 
    # hybrid[:,:,0] = instruction_im[:,:,0]+instruction_im[:,:,0]
    # hybrid[:,:,0] = hybrid[:,:,0] / hybrid[:,:,0].max()

    # find the maximum of either the stepper_driver and instchan per pixel
    hybrid[:,:,0] = np.maximum(instruction_im[:,:,0],instruction_im[:,:,0])

    hybrid[:,:,1] = power_im[:,:,0]
    hybrid[:,:,2] = power_im[:,:,1]

    plt.figure(6)
    plt.title("hybrid image")
    plt.imshow(hybrid)
    # plt.imsave(savePath + goodFiles[0]+"_"+"1"+"_"+"hybrid.png",hybrid)

#get hasp of all the data channels (1-4)
#it retuns a numpy array back.
def getRunHASP(run):
    dasps = np.zeros((256,256,4))
    # dasps[0] = run[0]
    # dasps[5] = run[5]
    # dasps[6] = run[6]
    for i in range(1,5):
        dasps[:,:,i-1] = getHASP(run[i],T)
    return dasps

#get hasp of all the data channels (1-4)
#it retuns a numpy array back.
def getRunFASP(run):
    dasps = np.zeros((256,256,4))
    # dasps[0] = run[0]
    # dasps[5] = run[5]
    # dasps[6] = run[6]
    for i in range(1,5):
        dasps[:,:,i-1] = getFASP(run[i],T)
    return dasps

#create images based on baseline and attack files
def createImages(goodFiles=[],attackFiles=[]):
    savePath = "./Data/images/fc_200_bw_500/"

    counter = 0

    for goodfile in goodFiles:
        with h5py.File(goodPath + goodfile, "r") as good1:
            print(goodfile)
            # showChannelDasp(good1["3"])
            for key in good1.keys():
                plt.imsave(savePath+"good/"+goodfile+"_"+key+"_"+str(good1[key].attrs['inc_label'])+".png",(getRunHASP(good1[key])//256).astype(np.uint8))
                counter += 1
                if(counter%100==0):
                    print(counter)

    for attackFile in attackFiles:
        with h5py.File(attackPath + attackFile, "r") as attack1:
            print(attackFile)
            # showChannelDasp(attack1["3"])
            for key in attack1.keys():
                plt.imsave(savePath+"bad/"+attackFile + "_" + key+"_"+str(attack1[key].attrs['inc_label'])+".png",(getRunHASP(attack1[key])//256).astype(np.uint8))
                counter += 1
                if(counter%100==0):
                    print(counter)

    # with h5py.File(attackPath + attackFiles[0], "r") as bad1:
        # showChannelDasp(bad1["200"])   

#this will use channel 6, inst_gt, to return an array
#of indicies that signify when an edge is found
def getEdges(chan_6):
    delta = chan_6[1:] - chan_6[:-1] #get the difference between two samples
    edges = np.abs(delta) >= 2.5           #Returns true only at indicies where there is an edge
    edges = np.nonzero(edges)[0]            #get indicies where there is an edge. Ignore the trailing [0], nonzero returns a tuple for some reason
    return edges

#window a sweep dataset and convert those windows into images
def createSweepImages():
    savePath = "./Data/images/sweep_data-HALF-BLUE-LAB-r10000d1800-091820-182807_1_fc_400_bw_800/"
    #because it is 10,000 KS/s, a window of 10000 is 1 second
    #every second, the speed increases by 1.
    sweepFile = sweepFiles[0]
    sweep1 = h5py.File(sweepPath + sweepFile, "r")
    run1 = sweep1["1"]
    #get each rising-falling edge.
    #These edges represent an increase in frequency
    edges = getEdges(run1[6])
    counter = 0
    for i in range(0,edges.shape[0]-1):
        e1 = edges[i]
        e2 = edges[i+1]
        plt.imsave(savePath+sweepFile + "_" + "1"+"_"+str(counter)+".png",(getRunHASP(run1[:,e1:e2])//256).astype(np.uint8))
        counter += 1
        if(counter%100==0):
            print(counter)

#create a DASP for each instruction and label them into a csv
def createInstructionImages():
    savePath = "./Data/images/random/"
    instructions = ["FWD-300-1024","REV-300-1024","FWD-500-2048","REV-1000-2048","IDLE-0-1000","FWD-700-200","REV-700-200","IDLE-0-2000"]
    goodFile = goodFiles[0]
    counter = 0
    filename = ""
    
    for goodFile in goodFiles:
        labelList = [] #will be used to be converted into a pandasdataframe for saving as a csv
        with h5py.File(goodPath + goodFile, "r") as good1:
            for key in good1.keys():
                run = good1[key]
                edges = getEdges(run[6])
        
                filename = goodFile + "_" + key +"_"+instructions[0]+".png"
                inst,step,numStep = instructions[0].split("-")
                labelList.append([filename,inst,step,numStep])
                #get beginning to the first edge
                plt.imsave(savePath+filename,(getRunHASP(run[:,:edges[0]])//256).astype(np.uint8))
                # print(key)
                # print(edges.shape)
                # print(edges)
                # print()
        
                if(edges.shape[0] == 7 or edges.shape[0] == 8):
                    for i in range(0,edges.shape[0]-1):
                        e1 = edges[i]
                        e2 = edges[i+1]
            
                        filename = goodFile + "_" + key +"_"+instructions[i+1]+".png"
                        inst,step,numStep = instructions[i+1].split("-")
                        labelList.append([filename,inst,step,numStep])            
            
                        plt.imsave(savePath+filename,(getRunHASP(run[:,e1:e2])//256).astype(np.uint8))
            
                    #get last edge to the end
                    filename = goodFile + "_" + key +"_"+instructions[-1]+".png"
                    inst,step,numStep = instructions[-1].split("-")
                    labelList.append([filename,inst,step,numStep])            
                    
                    plt.imsave(savePath+filename,(getRunHASP(run[:,edges[6]:])//256).astype(np.uint8))
            
            
                    counter += 1
                    if(counter%100==0):
                        print(counter)
                else:
                    print("could not get the edges on ",key,"number of edges found:",edges.shape)
        df = pd.DataFrame(labelList, columns = ['filename', 'instruction',"steps","number of steps"])
        print(df)
        df.to_csv("instruction_"+goodFile+".csv",index=False)




#create a DASP for each instruction and label them into a csv
def createInstructionImages_forRandom():
    savePath = "./Data/images/random_val/"
    counter = 0
    filename = ""
    
    randomFile = randomFiles[1]
    labelList = [] #will be used to be converted into a pandasdataframe for saving as a csv
    with h5py.File(randomPath + randomFile, "r") as random1:
        for key in random1.keys():
            run = random1[key]
            edges = getEdges(run[6])

            #remove these characters from the string
            instructions = run.attrs["instructions"].replace("'","")
            instructions =instructions.replace("[","")
            instructions =instructions.replace("]","")

            #convert the comma seperaction into dash seperation
            instructions = instructions.replace(", ","-")
            instructions = instructions.split("|")
            #Add a 0 to idle instruction to align with the three parameters of all other instructions
            for i in range(len(instructions)):
                if "idle" in instructions[i]:
                    instruction = instructions[i].split("-")
                    instructions[i] = instruction[0]+"-0-"+instruction[1]


    
            filename = randomFile + "_" + key +"_"+instructions[0]+".png"
            inst,step,numStep = instructions[0].split("-")
            labelList.append([filename,inst,step,numStep])
            #get beginning to the first edge
            plt.imsave(savePath+filename,(getRunHASP(run[:,:edges[0]])//256).astype(np.uint8))

            # print(key)
            print(edges.shape)
            # print(edges)
            # print()
    
            if(edges.shape[0] == 9 or edges.shape[0] == 10):
                for i in range(0,edges.shape[0]-1):
                    e1 = edges[i]
                    e2 = edges[i+1]
        
                    filename = randomFile + "_" + key +"_"+instructions[i+1]+".png"
                    inst,step,numStep = instructions[i+1].split("-")
                    labelList.append([filename,inst,step,numStep])            
        
                    plt.imsave(savePath+filename,(getRunHASP(run[:,e1:e2])//256).astype(np.uint8))

        
                #get last edge to the end
                filename = randomFile + "_" + key +"_"+instructions[-1]+".png"
                inst,step,numStep = instructions[-1].split("-")
                labelList.append([filename,inst,step,numStep])            
                
                plt.imsave(savePath+filename,(getRunHASP(run[:,edges[6]:])//256).astype(np.uint8))

        
        
                counter += 1
                if(counter%100==0):
                    print(counter)
            else:
                print("could not get the edges on ",key,"number of edges found:",edges.shape)
        df = pd.DataFrame(labelList, columns = ['filename', 'instruction',"steps","number of steps"])
        print(df)
        df.to_csv("instruction_"+randomFile+".csv",index=False)






if __name__ == '__main__':
    plt.close("all")

    titles = ["index","stepper_driver", "ps_wall","driver_ps","instchan","run_gt","inst_gt"]


    sweepPath = "./Data/frequency_sweep/"
    attackPath = "./Data/simulated_attack/known-bad/"
    goodPath = "./Data/simulated_attack/baseline-known-good/"
    randomPath = "./Data/random/"

    savePath = "./Data/images/tests/"

    sweepFiles = ["data-HALF-BLUE-LAB-r10000d1800-091820-182807.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-142849.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-154000.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-170842.h5f"]
    goodFiles = ["data-increment-BL-r10000d18000-051320-125438.h5f","data-increment-BL-r10000d18000-051320-184456.h5f","data-increment-BL-r10000d18000-051420-110816.h5f","data-increment-BL-r10000d18000-051420-171038.h5f"]
    attackFiles = ["data-increment-MUTSPD5-r10000d18000-061520-105843.h5f","data-increment-MUTSPD5-r10000d18000-061620-002749.h5f","data-increment-MUTSPD-r10000d18000-051720-150804.h5f","data-increment-MUTSPD-r10000d18000-051820-114805.h5f","data-increment-MUTSPD-r10000d18000-052920-153932.h5f"]
    randomFiles = ["data-random-inst-r10000d7200-070821-131756.h5f","data-eval-rand-r40000d7200-072121-134715.h5f"]
    

    #the sampling rate is 10kS/s
    # T = 10000 #for toy system
    T = 40000 #for eval system
    F = 1/T

    # createImages(goodFiles,attackFiles)
    # createSweepImages()
    # createInstructionImages()
    createInstructionImages_forRandom()

     

