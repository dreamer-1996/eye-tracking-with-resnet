# Gaze tracking using off-the-shelf webcam employing Deep Learning techniques

[Here](https://github.com/hysts/pytorch_mpiigaze_demo) is a demo program.
Following repo is modified from this [work](https://github.com/hysts/pytorch_mpiigaze).


## Requirements

* Python >= 3.7

```bash
pip install -r requirements.txt
```

## Instructions for training and evaluating the model.

Download the `dataset.zip` file from [here](https://drive.google.com/file/d/1jJMUZ8wvEEs8q3lqGGa3gxwGA8syy2OO/view?usp=sharing).

This is different compared to the original MPIIGaze dataset because here we have added the 2-D points on the screen to the normalized data.Our target is to find the 2D gaze points on the screen from images and head poses.

Extract the `Data` folder and place it inside `datasets/MPIIGaze`

Then preprocess it :

```bash
python3 tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/
```
Then run the script :

```bash
scripts/run_all_mpiigaze_resnet_preact.sh

```
The results, checkpoints and the logs will be generated automatically inside the `experiments` folder.

After training and evaluation is complete, take the last checkpoint file for any experiment for example take the 4th checkpoint .pth file from 
`experiments/mpiigaze/resnet_preact/exp00/00` and place it in `eye-tracking-with-resnet/data/models/mpiigaze/resnet_preact`

Then run the script to download the dlib face model:
```bash
scripts/download_dlib_model.sh
```
Now run the demo:
```bash
python3 demo.py --config configs/demo_mpiigaze_resnet.yaml
```


## Further work needed

* Camera calibration is needed. Save the calibration result in the same format as the sample file `data/calib/sample_params.yaml`.
* A system should be created such that the `model.pth` gets retrained in real time with some images of the person whose gaze is to be tracked next. This provides the model with some information of the next person whose gaze is to be tracked. This increases the accuracy of the model and we have verified that with experiments.
* Trying with different CNN, optimizing the parameters to see if any change in accuracy is happening.




## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
* Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)



