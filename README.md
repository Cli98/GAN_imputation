# GAN_imputation
### Missing Gene data imputation by GAN

### Here is an outline for this repo

#### --- Preprocess yeast genotype to required format. Add noise of binormal distribution to generate corrupted data.
#### --- Use DCGAN and GAIN and denoising autoencoder as the model to train data separately, compare their performance.
#### --- Under the missing_ratio of 5%,10%,20%, the GAN model has a test period performance(in terms of accuracy) of 91%-93%. This is pretty strong compared with average of row(48%), SVR (65%) and knn(73%) method. However, AE get a better performance(99%) on average.
