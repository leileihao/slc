Here contains the GitHub repository for the code associated with the publication "Heart rate and sleep history encode ultradian REM sleep timing."

## Sleep Classification Model
### Getting started

Each of the data folder in the training set is a recording of a mouse at a particular time. Let's use the example folder with the name `AC81_081221n1`. The important files from each folder needed to test the model include:

* AC81_081221n1/`EEG.mat`
* AC81_081221n1/`EMG.mat`

If you desire to train your own model, you will also need the additional `remidx_AC81_081221n1.txt`. These are the manual annotations used to train the model.

### Model basics

SleepClass is a two-stage model, see the model architecture below. For pre-processing, EEG and EMG signals are downsampled and normalized for each recording using z-scoring. 2.5-s bins of EEG and EMG data are concatenated. In stage 1, 2.5-s bins of EEG/EMG data are encoded by a BiLSTM model and then mapped by fully connected (FC) layers (FC1 classifier) into a 3D latent feature space. Finally, each predicted state (point) in the latent space is mapped into REM, NREM, or wake using the argmax function. 

To improve the prediction, Stage 2 incorporates the temporal context of each 2.5-s bin, by combining the latent space feature from Stage 1 for time point t with the 25 preceding and following feature bins. The combined latent space features are fed through a feedforward network mapping them into a 3D output vector, which is finally transformed using argmax into a single label indicating REM, NREM, or wake. 

### Use the model

To directly use the model, run the `SleepClassTest.ipynb` file. The previously trained models from the sample data (https://drive.google.com/drive/folders/1upqAJ86dXT-J-EcccTPbD95Dq0KMcEbo?usp=sharing) is saved as a `.pickle` file.

* Stage 1 trained model: `MultiVanillaModelPart1N.pkl`
* Stage 2 trained model: `MultiVanillaModelPart2Ov2.pkl`

An additional packages that may be required include:
* sleepy.py: https://github.com/tortugar/Lab/tree/master/PySleep

### Train the model

Alternatively, you can use your own data to train the model. The code for training each stage is separately included in `model_stage1.ipynb` and `model_stage2.ipynb`. The outputs from model_stage1.ipynb is then used in the code of `model_stage2.ipynb` to train the second stage. The only only changes that need to be made are the input data for training and any differences in sampling rate or additional model tuning.


## Time Until Next REM Model

This is a random forest model that extracts the latent space of the sleep classifcation model and uses it to predict when the next REM episode will occur. The code to load each recording, run the recording through the sleep classification stage 1 model, train, and test the random forest is in the `time_until_next_rem.ipynb` file.

