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
- Sravan 

# CHAPTER 3: Deploy Process On VCK5000
- rahul 

# CHAPTER 4: RESULTS
- vedant 

# CHAPTER 5: Future Goals
# CHAPTER 6: Conclusion 