#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:47:01 2019

@author: seykia
"""
# ------------------------------------------------------------------------------
#  Usage:
#  python normative_NP.py -r /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/responses.nii.gz 
#                           -c /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/covariates.pickle 
#                           --tr /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_responses.nii.gz 
#                           --tc /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/test_covariates.pickle 
#                           -o /home/preclineu/andmar/data/seykia/ds000030_R1.0.5/Results
#
#
#  Written by S. M. Kia
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import sys
import argparse
import torch
from torch import optim
import numpy as np
import pickle
from NP import NP, apply_dropout_test, np_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from architecture import Encoder, Decoder
from nispat.utils import compute_pearsonr, explained_var, compute_MSLL
from nispat.utils import extreme_value_prob, extreme_value_prob_fit, ravel_2D, unravel_2D
from nispat import fileio

 

def get_args(*args):
    """ Parse command line arguments"""

    ############################ Parsing inputs ###############################
    
    parser = argparse.ArgumentParser(description='Neural Processes (NP) for Deep Normative Modeling')
    parser.add_argument("-r", help="Training response pickle file address", 
                        required=True, dest="respfile", default=None)
    parser.add_argument("-c", help="Training covariates pickle file address", 
                        required=True, dest="covfile", default=None)
    parser.add_argument("--tc", help="Test covariates pickle file address", 
                        required=True, dest="testcovfile", default=None)
    parser.add_argument("--tr", help="Test response pickle file address", 
                        required=True, dest="testrespfile", default=None)
    parser.add_argument("-o", help="Output directory address", dest="outdir", default=None)
    parser.add_argument('-m', type=int, default=10, dest='m',
                        help='number of fixed-effect estimations')
    parser.add_argument('--batchnum', type=int, default=10, dest='batchnum',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs',
                        help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', dest='device',
                        help='Either cpu or cuda')
    parser.add_argument('--fxestimator', type=str, default='MT', dest='estimator',
                        help='Fixed-effect estimator type.')

    args = parser.parse_args()


    cuda = args.device=='cuda' and torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda else "cpu")
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    return args
    
def estimate(args):
    print('Loading the input Data ...')
    responses = fileio.load_nifti(args.respfile, vol=True).transpose([3,0,1,2])
    with open(args.covfile, 'rb') as handle:
        covariates = pickle.load(handle)['covariates']
    test_responses = fileio.load_nifti(args.testrespfile, vol=True).transpose([3,0,1,2])
    with open(args.testcovfile, 'rb') as handle:
        test_covariates = pickle.load(handle)['test_covariates']
    
    torch.set_default_dtype(torch.float32)
       
    mask = fileio.create_mask(responses[0,:,:,:], mask=None)
    
    response_shape = responses.shape
    test_responses_shape = test_responses.shape
    
    print('Normalizing the input Data ...')
    covariates_scaler = StandardScaler()
    covariates = covariates_scaler.fit_transform(covariates)
    test_covariates = covariates_scaler.transform(test_covariates)
    response_scaler = MinMaxScaler()
    responses = unravel_2D(response_scaler.fit_transform(ravel_2D(responses)), response_shape)
    test_responses = unravel_2D(response_scaler.transform(ravel_2D(test_responses)), test_responses_shape)
    
    factor = args.m
    
    x_context = np.zeros([covariates.shape[0], factor, covariates.shape[1]], dtype=np.float32)
    y_context = np.zeros([responses.shape[0], factor, responses.shape[1], 
                         responses.shape[2], responses.shape[3]], dtype=np.float32)
    x_all = np.zeros([covariates.shape[0], factor, covariates.shape[1]], dtype=np.float32)
    y_all = np.zeros([responses.shape[0], factor, responses.shape[1], 
                         responses.shape[2], responses.shape[3]], dtype=np.float32)
    x_context_test = np.zeros([test_covariates.shape[0], factor, test_covariates.shape[1]], dtype=np.float32)
    y_context_test = np.zeros([test_responses.shape[0], factor, test_responses.shape[1], 
                         test_responses.shape[2], test_responses.shape[3]], dtype=np.float32)
    
    print('Estimating the fixed-effects ...')
    for i in range(factor):
        x_context[:,i,:] = covariates[:,:]
        x_context_test[:,i,:] = test_covariates[:,:]
        idx = np.random.randint(0,covariates.shape[0], covariates.shape[0])
        if args.estimator=='ST':
            for j in range(responses.shape[1]):
                for k in range(responses.shape[2]):
                    for l in range(responses.shape[3]):
                        reg = LinearRegression()
                        reg.fit(x_context[idx,i,:], responses[idx,j,k,l])
                        y_context[:,i,j,k,l] = reg.predict(x_context[:,i,:])
                        y_context_test[:,i,j,k,l] = reg.predict(x_context_test[:,i,:])
        elif args.estimator=='MT':
            reg = MultiTaskLasso(alpha=0.1)
            reg.fit(x_context[idx,i,:], np.reshape(responses[idx,:,:,:], [covariates.shape[0],np.prod(responses.shape[1:])]))
            y_context[:,i,:,:,:] = np.reshape(reg.predict(x_context[:,i,:]), 
                     [x_context.shape[0],responses.shape[1],responses.shape[2],responses.shape[3]])    
            y_context_test[:,i,:,:,:] = np.reshape(reg.predict(x_context_test[:,i,:]), 
                          [x_context_test.shape[0],responses.shape[1],responses.shape[2],responses.shape[3]])
        print('Fixed-effect %d of %d is computed!' %(i+1, factor))
    
    x_all = x_context
    y_all = np.expand_dims(responses, axis=1).repeat(factor, axis=1)
    y_test = np.expand_dims(test_responses, axis=1)
    del responses, test_responses
    
    ################################## TRAINING #################################  
    
    encoder = Encoder(x_context, y_context, args).to(args.device)
    args.cnn_feature_num = encoder.cnn_feature_num
    decoder = Decoder(x_context, y_context, args).to(args.device)
    model = NP(encoder, decoder, args).to(args.device)
    
    k = 1
    epochs = [int(args.epochs/4),int(args.epochs/2),int(args.epochs/5),int(args.epochs-args.epochs/4-args.epochs/2-args.epochs/5)]
    mini_batch_num = args.batchnum
    batch_size = int(x_context.shape[0]/mini_batch_num)
    model.train()
    for e in range(len(epochs)): 
        optimizer = optim.Adam(model.parameters(), lr=10**(-e-2))
        for j in range(epochs[e]):
            train_loss = 0
            rand_idx = np.random.permutation(x_context.shape[0])
            for i in range(mini_batch_num):
                optimizer.zero_grad()
                idx = rand_idx[i*batch_size:(i+1)*batch_size]
                y_hat, z_all, z_context, dummy = model(torch.tensor(x_context[idx,:,:], device = args.device), 
                                                   torch.tensor(y_context[idx,:,:,:,:], device = args.device), 
                                                   torch.tensor(x_all[idx,:,:], device = args.device), 
                                                   torch.tensor(y_all[idx,:,:,:,:], device = args.device))
                loss = np_loss(y_hat, torch.tensor(y_all[idx,:,:,:,:], device = args.device), z_all, z_context)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch: %d, Loss:%f' %(k, train_loss))
            k += 1
            
    ################################## Evaluation #################################
    
    model.eval()
    model.apply(apply_dropout_test)
    with torch.no_grad():
        y_hat, z_all, z_context, y_sigma = model(torch.tensor(x_context_test, device = args.device),
                                                 torch.tensor(y_context_test, device = args.device), n = 100)
        test_loss = np_loss(y_hat, torch.tensor(y_test, device = args.device), z_all, z_context).item()
        print('Test loss:%f' %(test_loss))
         
    RMSE = np.sqrt(np.mean((y_test - y_hat.cpu().numpy())**2, axis = 0)).squeeze() * mask
    SMSE = RMSE ** 2 / np.var(y_test, axis=0).squeeze()
    Rho, pRho = compute_pearsonr(y_test.squeeze(), y_hat.cpu().numpy().squeeze())
    EXPV = explained_var(y_test.squeeze(), y_hat.cpu().numpy().squeeze()) * mask
    MSLL = compute_MSLL(y_test.squeeze(), y_hat.cpu().numpy().squeeze(), 
                        y_sigma.cpu().numpy().squeeze()**2, train_mean = y_test.mean(0), 
                        train_var = y_test.var(0)).squeeze() * mask
                        
    NPMs = (y_test - y_hat.cpu().numpy()) / (y_sigma.cpu().numpy())
    NPMs = NPMs.squeeze()
    NPMs = NPMs * mask
    NPMs = np.nan_to_num(NPMs)
    
    temp=NPMs.reshape([NPMs.shape[0],NPMs.shape[1]*NPMs.shape[2]*NPMs.shape[3]])
    EVD_params = extreme_value_prob_fit(temp, 0.01)
    abnormal_probs = extreme_value_prob(EVD_params, temp, 0.01)
    
    ############################## SAVING RESULTS #################################
    
    if args.outdir is not None:
        print('Saving Results to: %s' %(args.outdir))
        exfile = args.testrespfile
        y_hat = y_hat.squeeze().cpu().numpy()
        y_hat = response_scaler.inverse_transform(ravel_2D(y_hat))
        y_hat = y_hat[:,mask.flatten()]
        fileio.save(y_hat.T, args.outdir + 
                    '/yhat.nii.gz', example=exfile, mask=mask)
        ys2 = y_sigma.squeeze().cpu().numpy()
        ys2 = response_scaler.inverse_transform(ravel_2D(ys2))
        ys2 = ys2**2
        ys2 = ys2[:,mask.flatten()]
        fileio.save(ys2.T, args.outdir + 
                    '/ys2.nii.gz', example=exfile, mask=mask) 
        NPMs = ravel_2D(NPMs)[:,mask.flatten()]
        fileio.save(NPMs.T, args.outdir +  
                    '/Z.nii.gz', example=exfile, mask=mask)
        fileio.save(Rho.flatten()[mask.flatten()], args.outdir +  
                    '/Rho.nii.gz', example=exfile, mask=mask)
        fileio.save(pRho.flatten()[mask.flatten()], args.outdir +  
                    '/pRho.nii.gz', example=exfile, mask=mask)
        fileio.save(RMSE.flatten()[mask.flatten()], args.outdir +  
                    '/rmse.nii.gz', example=exfile, mask=mask)
        fileio.save(SMSE.flatten()[mask.flatten()], args.outdir +  
                    '/smse.nii.gz', example=exfile, mask=mask)
        fileio.save(EXPV.flatten()[mask.flatten()], args.outdir +  
                    '/expv.nii.gz', example=exfile, mask=mask)
        fileio.save(MSLL.flatten()[mask.flatten()], args.outdir +  
                    '/msll.nii.gz', example=exfile, mask=mask)
        
        with open(args.outdir +'model.pkl', 'wb') as handle:
             pickle.dump({'model':model, 'reg':reg, 'covariates_scaler':covariates_scaler,
                          'response_scaler': response_scaler, 'EVD_params':EVD_params,
                          'abnormal_probs':abnormal_probs}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
###############################################################################
    print('DONE!')


def main(*args):
    """ Parse arguments and estimate model
    """
    
    np.seterr(invalid='ignore')
    args = get_args(args)
    estimate(args)
    
if __name__ == "__main__":
    main(sys.argv[1:])
