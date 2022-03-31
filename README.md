# Deepfake_Inference_On_VCK5000

# CHAPTER 1: INTRODUCTION

## 1.1 Abstract:
Better generative models and larger datasets have led to more realistic fake videos that can fool the human eye and machines. Today, the danger of fake news is widely acknowledged, and in a context where more than 100 million hours of video content are watched daily on social networks, the spread of falsified video raises more and more concerns. While significant improvements have been made in the field of deepfakes classification, deepfakes detection and reconstruction have remained a difficult task. Because of the rapid increase in technology, in the future deep fake videos can be seen everywhere, for example, on live news channels. So it is required to effectively detect the deepfake in the least time and possibly restore it. In this report, we are presenting the initial results of our experiments on forgery segmentation. Also, we have performed various experiments of running Quantized Neural Networks (QNNs) on FPGAs for better interference.

## 1.2 Technical Keywords:

Deepfakes, Image Forensics, CNN, Image Reconstruction, Image Augmentations, FPGA, QNN

## 1.3 Problem Statement:

### DEEPFAKE D-I-R (Detection-Inference-Restoration)

**Detection:** Identifying manipulated regions by creating a segmentation mask and improving detections using the mask.
**Inference:** Utilizing FPGA hardware-based acceleration for reducing inference time and power requirements compared to traditional methods.
**Restoration:** Restoring the original face of the person on whom the deepfake video was created.

## 1.4 Hardware Setup:
We had tried setting up the VCK5000 on various system, following are the details about the setups that we had tried:
![](assets/2022-03-01-11-58-15.png)

Finally, we have been continuously running the VCK5000 Card on the Dell R740 Server. Here are some Images attached of our final setup
![](assets/2022-03-01-11-58-42.png) 

Note 1: Guidelines for the VCK5000 Card Installation Can be found here: https://www.xilinx.com/member/vck5000-aie.html (This is asecured site, you need to request for the same)
Note 2: We have used external SMPS for the VCK5000 AUX Power Supply. The R740 Dell Server didn't have the same connector. <br>
Note 3: Following Aliases, we often use to validate and check the temperature of VCK5000.

## 1.5 VCK5000 Software Setup:

After the Hardware Installation first check whether the card is getting detected or not using the following command: 

`lspci -vd 10ee:`

The output should be similar to like this:
```
02:00.0 Processing accelerators: Xilinx Corporation Device 5044

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 16, NUMA node 0

        Memory at 380030000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038020000 (64-bit, prefetchable) [size=128K]

        Capabilities: <access denied>

        Kernel driver in use: xclmgmt

        Kernel modules: xclmgmt

02:00.1 Processing accelerators: Xilinx Corporation Device 5045

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 17, NUMA node 0

        Memory at 380038000000 (64-bit, prefetchable) [size=128K]

        Memory at 380028000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038040000 (64-bit, prefetchable) [size=64K]

        Capabilities: <access denied>

        Kernel driver in use: xocl

        Kernel modules: xocl
```

Afterward, follow these steps:

```
#The Kernel We are using Right Now:
vedant@vck5000:~$ uname -a
Linux vck5000 5.4.0-100-generic #113-Ubuntu SMP Thu Feb 3 18:43:29 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux 

# Clone the Vitis AI 1.4.1 Repo:
https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI/
git checkout 1.4.1 

# go to the setup directory of VCK5000:
cd setup/vck5000/

# Install the XRT and other modules using the script given in the repo
source ./install.sh

# Installl deb packages for the card
wget https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz -O xilinx-vck5000-es1-g en3x16-platform-2-1_all.deb.tar.gz
tar -xzvf xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz
sudo dpkg -i xilinx-sc-fw-vck5000_4.4.6-2.e1f5e26_all.deb 
sudo dpkg -i xilinx-vck5000-es1-gen3x16-validate_2-3123623_all.deb 
sudo dpkg -i xilinx-vck5000-es1-gen3x16-base_2-3123623_all.deb

# Flash the card:
sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan 
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update

# DO the cold reboot of the system and validate the card
sudo /opt/xilinx/xrt/bin/xbutil validate --device 0000:01:00.1
```

# CHAPTER 2: Software Model
## 2.1 U-YNet Model

There are many pre-trained models that perform deepfakes classification and localization tasks separately, but there are no models that do the task simultaneously. Such models where 2 tasks share the same backbone are called multi-task models.

We are using this architecture as the tasks performed by the multi-tasking models have the advantage of learning from each other; i.e., in our model, the classification task learns from the segmentation task, hence improving the accuracy and vice-versa. We call this model the U-YNet architecture as the 2 tasks share a UNet backbone.

![image](https://user-images.githubusercontent.com/22630228/161117542-5406d25f-2234-46f5-bec5-31524c919366.png)

## 2.2 Dataset

All currently available public DeepFake datasets include both real and manipulated videos and images. There are many datasets like UADFV, FaceForensics++, CelebDF, Google DFD and the DFDC dataset. Fake videos were generated from these recorded or collected videos using different DeepFake generators.

![image](https://user-images.githubusercontent.com/22630228/161117672-0f36a5b5-56d0-4f9c-8dfd-e9643f37eeb2.png)

As we can see from the above diagram DFDC is currently the largest available deepfakes dataset with the most number of videos and faces. So we decided to choose this dataset over others for training the model.

![image](https://user-images.githubusercontent.com/22630228/161117736-6b129048-f84b-47ea-9784-ffa1950b992e.png)

The dataset does not come with the segmentation masks that we need for training the segmentation branch of the U-YNet model. To create the mask, we generate a pixel-wise difference mask by calculating the Structural Similarity Index (SSIM) between the frame of a real, and its corresponding fake video. This difference mask contains 1 for manipulated pixels and 0 for real ones.

![image](https://user-images.githubusercontent.com/22630228/161117803-9fde9ab0-5c40-4e37-90da-a2039469588a.png)

One problem observed while training models is that, initially we were getting 99% accuracy on the DFDC dataset, which meant that the model was overfitting. After insinuating on this problem, we found the reason. While randomly splitting the dataset into testing, training, and validation datasets, the training dataset had examples of all the faces in the dataset. So the model was basically learning/remembering the faces and not the deepfake features. So, while testing the model on the testing dataset, the model was giving very high accuracy. This problem is called data leakage.

To solve this problem we used the information given in the metadata.json file that DFDC provides. metadata.json file provides the mapping of fake faces to the real faces from which they were produced. We split the dataset such that the testing data had no faces similar to those in the training dataset.

We will release the dataset as soon as possible.

# CHAPTER 3: Deploy Process On VCK5000
- rahul 

# CHAPTER 4: RESULTS
- vedant 

# CHAPTER 5: Future Goals
# CHAPTER 6: Conclusion 
