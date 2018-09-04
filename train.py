import tensorflow as tf
import numpy as np
import os, time

from cyclegan import CycleGAN
from preprocess import *

def train(input_A, input_B, g_type = g_type,n_epochs = n_epochs, n_features = n_features, n_frames = n_frames, log_dir = log_dir, model_dir = model_dir) :
    
    generator_lr = 0.0002
    generator_lr_decay = generator_lr/200000
    discriminator_lr = 0.0001
    discriminator_lr_decay = discriminator_lr/200000
    cycle_lambda = 10
    identity_lambda = 5 

    # Preprocessing datasets
    A_norm, B_norm = preprocess(input_A,input_B)
    
    # Cyclegan_voice convert
    model = CycleGAN(num_features=n_features, g_type=g_type, log_dir=log_dir)
    
    print("Start Training...")
    for epoch in range(n_epochs) :
        print("Epoch : %d " % epoch ) 
        start_time = time.time()
        train_A, train_B = sample_train_data(dataset_A=A_norm, dataset_B=B_norm,n_frames=n_frames) # random data
        
        n_samples = train_A.shape[0]
        
        for i in range(n_samples) : # mini_ batch_size = 1
            n_iter = n_samples * epoch + i
            
            if n_iter > 10000 :
                identity_lambda = 0
            if n_iter > 200000 :
                generator_lr = max(0, generator_lr - generator_lr_decay)
                discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
            
            start = i
            end = start + 1
            
            generator_loss, discriminator_loss = model.train(input_A = train_A[start:end], 
                                                             input_B = train_B[start:end], 
                                                             cycle_lambda = cycle_lambda,
                                                             identity_lambda = identity_lambda,
                                                             generator_lr = generator_lr,
                                                             discriminator_lr = discriminator_lr)
        end_time = time.time()
        epoch_time = end_time-start_time
        print("Generator Loss : %f, Discriminator Loss : %f, Time : %02d:%02d" % (generator_loss, discriminator_loss,(epoch_time % 3600 // 60),(epoch_time % 60 // 1)))
        model.save(directory = model_dir, filename = "model")

if __name__ == "__main__" :
    train(input_A = dataset_A, input_B = dataset_B)
    print("Training Done!")
            
            