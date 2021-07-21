# Gaze tracking using off-the-shelf webcam employing Deep Learning techniques

[Here](https://github.com/hysts/pytorch_mpiigaze_demo) is a demo program.


## Requirements

* Linux (Tested on Ubuntu only)
* Python >= 3.7

```bash
pip install -r requirements.txt
```


## Download the dataset and preprocess it

### MPIIGaze

Download the dataset zip file from here :
Extract the `Data` folder and place it inside `datasets/MPIIGaze`
Then preprocess it :

```bash
python3 tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/
```


```


## Usage

Using [`scripts/run_all_mpiigaze_resnet_preact.sh`](scripts/run_all_mpiigaze_resnet_preact.sh),you can run all training and evaluation of ResNet-8 with default parameters.





## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
* Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)



