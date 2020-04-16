#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from skimage import img_as_ubyte


# In[2]:


plt.close('all')
clear = lambda: os.system('clear')
clear()
np.random.seed(110)

colors = [[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5],[0.5,0,0.5]] #List of colours

imgNames = ['water_coins','jump','tiger']#{'balloons', 'mountains', 'nature', 'ocean', 'polarlights'};
segmentCounts = [2,3,4,5]


# In[3]:


img_num = np.zeros((len(imgNames)*len(segmentCounts)),dtype='int') #stores the last iteration number before convergence of EM
itr_cnt = -1 #iteration counter to traverse img_num

for imgName in imgNames:
    for SegCount in segmentCounts:
        itr_cnt+=1 #increment iteration counter
        # Load the imageusing MatPlotLib        
        img_mtlb = mpimg.imread("Input/" + imgName+ ".png")
        print('Using Matplotlib Image Library: Image is of datatype ',img_mtlb.dtype,'and size ',img_mtlb.shape) # Image is of type float          
        
        # Load the Pillow-- the Python Imaging Library
        img = Image.open("Input/" + imgName+ ".png").convert('RGB')
        print('Using Pillow (Python Image Library): Image is of datatype ',img.info,'and size ',img.size) # Image is of type uint8  
        
        
        #%% %Define Parameters
        nSegments = SegCount   # of color clusters in image
        nPixels = img_mtlb.shape[0]*img_mtlb.shape[1];    # Image can be represented by a matrix of size nPixels*nColors
        maxIterations = 20; #maximum number of iterations allowed for EM algorithm.
        nColors = 3;
        #%% Determine the output path for writing images to files
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']));
        if not(os.path.exists(outputPath)):
            os.makedirs(outputPath)
        mpimg.imsave(''.join([outputPath,'0.png']),img_mtlb) #save using Matplotlib image library
        #%% Vectorizing image for easier loops- done as im(:) in Matlab
        pixels = img
        pixels = np.array(img).reshape(nPixels,nColors,1)
        
        #%%
        """ Initialize pi (mixture proportion) vector and mu matrix (containing means of each distribution)
            Vector of probabilities for segments... 1 value for each segment.
            Best to think of it like this...
            When the image was generated, color was determined for each pixel by selecting
            a value from one of "n" normal distributions. Each value in this vector 
            corresponds to the probability that a given normal distribution was chosen."""
        
        
        """ Initial guess for pi's is 1/nSegments. Small amount of noise added to slightly perturb 
           GMM coefficients from the initial guess"""
           
        pi = 1/nSegments*(np.ones((nSegments, 1),dtype='float'))
        increment = np.random.normal(0,.0001,1)
        for seg_ctr in range(len(pi)):
            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
                if pi[seg_ctr] > 1:
                    pi[seg_ctr] = 1
            else:
                pi[seg_ctr] = pi[seg_ctr] - increment
                if pi[seg_ctr] < 0:
                    pi[seg_ctr] = 0
                
         #%% 
        """Similarly, the initial guess for the segment color means would be a perturbed version of [mu_R, mu_G, mu_B],
           where mu_R, mu_G, mu_B respectively denote the means of the R,G,B color channels in the image.
           mu is a nSegments X nColors matrrix,(seglabels*255).np.asarray(int) where each matrix row denotes mean RGB color for a particcular segment"""
           
        mu = 1/nSegments*(np.ones((nSegments, nColors),dtype='float'))  #for even start
        #add noise to the initialization (but keep it unit)
        for seg_ctr in range(nSegments):
            if(seg_ctr%2==1):
                increment = np.random.normal(0,.0001,1)
            for col_ctr in range(nColors):
                if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) - increment;             
        

        #%% EM-iterations begin here. Start with the initial (pi, mu) guesses        
        
        mu_last_iter = mu;
        pi_last_iter = pi;
        
        
        for iteration in range(maxIterations):
            img_num[itr_cnt] = iteration
            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % -----------------   E-step  -----estimating likelihoods and membership weights (Ws)
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' E-step']))
            # Weights that describe the likelihood that pixel denoted by "pix_import scipy.miscctr" belongs to a color cluster "seg_ctr"
            Ws = np.ones((nPixels,nSegments),dtype='float')  # temporarily reinitialize all weights to 1, before they are recomputed

            """ logarithmic form of the E step."""
            
            for pix_ctr in range(nPixels):
                # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float')
                for seg_ctr in range(nSegments):
                    x_minus_mu_T  = np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu    = ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr]) - .5*(np.dot(x_minus_mu_T,x_minus_mu))
                
                # Note the max
                logAmax = max(logAjVec.tolist()) 
                
                # Calculate the third term from the final eqn in the above link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)
                
                # Here Ws are the relative membership weights(p_i/sum(p_i)), but computed in a round-about way 
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm)
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)
                

            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % -----------------   M-step  --------------------
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
            
            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' M-step: Mixture coefficients']))
            #%% temporarily reinitialize mu and pi to 0, before they are recomputed
            mu = np.zeros((nSegments,nColors),dtype='float') # mean color for each segment
            pi = np.zeros((nSegments,1),dtype='float') #mixture coefficients

            
            for seg_ctr in range(nSegments):
                '''
                denominatorSum = 0;
                for pix_ctr in range(nPixels):
                    mu[seg_ctr] = mu[seg_ctr] + pixels[pix_ctr,:,0]*Ws[pix_ctr,seg_ctr]
                    denominatorSum = denominatorSum + Ws[pix_ctr][seg_ctr]
                '''
                denominatorSum = np.sum(Ws[:,seg_ctr])
                mu[seg_ctr] = np.sum(np.multiply(pixels[:,:,0],np.tile(np.reshape(Ws[:,seg_ctr],(Ws[:,seg_ctr].shape[0],1)),(1,3))),axis=0)
                
                ## Update mu
                mu[seg_ctr,:] =  mu[seg_ctr,:]/ denominatorSum;
                ## Update pi
                pi[seg_ctr] = denominatorSum / nPixels; #sum of weights (each weight is a probability) for given segment/total num of pixels   
        

            print(np.transpose(pi))

            muDiffSq = np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))

            if (muDiffSq < .0000001 and piDiffSq < .0000001): #sign of convergence
                print('Convergence Criteria Met at Iteration: ',iteration, '-- Exiting code')
                break;
            

            mu_last_iter = mu;
            pi_last_iter = pi;


            ##Draw the segmented image using the mean of the color cluster as the 
            ## RGB value for all pixels in that cluster.
            segpixels = np.array(pixels)
            cluster = 0
            for pix_ctr in range(nPixels):
                cluster = np.where(Ws[pix_ctr,:] == max(Ws[pix_ctr,:]))
                vec     = np.squeeze(np.transpose(mu[cluster,:])) 
                segpixels[pix_ctr,:] =  vec.reshape(vec.shape[0],1)
            
            
            segpixels = np.reshape(segpixels,(img_mtlb.shape[0],img_mtlb.shape[1],nColors)) ## reshape segpixels to obtain R,G, B image
            segpixels = img_as_ubyte(segpixels)
            segpixels = rgb2gray(segpixels)
            
            kmeans = KMeans(n_clusters = SegCount).fit(np.reshape(segpixels,(nPixels, 1)))
            seglabels = np.reshape(kmeans.labels_, (img_mtlb.shape[0], img_mtlb.shape[1]))
            seglabels = gaussian(np.clip(label2rgb(seglabels,colors= colors), 0,1), sigma = 2, multichannel = False)
            
            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabels) #save the segmented output


# In[4]:


# Displaying final segmented outputs
itr_cnt = 0
fig = plt.figure(figsize = (40,30))
for imgName in imgNames:
    for SegCount in segmentCounts:
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']))
        img = mpimg.imread(outputPath + str(img_num[itr_cnt]) + ".png")
        itr_cnt+=1
        a = fig.add_subplot(3,4,itr_cnt)
        a.set_title(imgName + " " + str(SegCount),fontsize=32)
        plt.imshow(img)


# In[ ]:




