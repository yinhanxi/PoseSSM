# [Sequential Joint Dependency Aware Human Pose Estimation with State Space Model (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33029)

## Dataset setup
Please download the h36m dataset [here](https://drive.google.com/drive/folders/1gNs5PrcaZ6gar7IiNZPNh39T7y6aPY3g). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Dependencies

- Python 3.10.9 
- PyTorch 1.12.1

```sh
pip install -r requirement.txt
```
## Pretrained model
The checkpoints pretrained with CPN and GT 2D joints as input are located in the `ckpt` folder.

## Test the model
To test on Human3.6M with 2D poses detected by CPN as inputs, run:

```
python main.py --reload --keypoints cpn_ft_h36m_dbb --previous_dir "ckpt/cpn" 
```
To test on Human3.6M with GT 2D poses as inputs, run:

```
python main.py --reload --keypoints gt --previous_dir "ckpt/gt" 
```

## Train the model
To train on Human3.6M with 2D poses detected by CPN as inputs, run:

```
python main.py --train --keypoints cpn_ft_h36m_dbb -n 'your_exp_name'
```
To train on Human3.6M with GT 2D poses as inputs, run:

```
python main.py --train --keypoints gt -n 'your_exp_name'
```

## Acknowledgement
Our code is extended from [DC-GCT](https://github.com/KHB1698/DC-GCT) and part of the code is borrowed from [Mamba](https://github.com/state-spaces/mamba).
We thank the authors for releasing the codes.
