## Dataset setup
Please download the dataset [here](https://drive.google.com/drive/folders/1gNs5PrcaZ6gar7IiNZPNh39T7y6aPY3g). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Test the model
To test on Human3.6M with detected 2D poses by CPN as inputs, run:

```
python main.py --reload --keypoints cpn_ft_h36m_dbb --previous_dir "ckpt/cpn" 
```
To test on Human3.6M with GT 2D poses as inputs, run:

```
python main.py --reload --keypoints gt --previous_dir "ckpt/gt" 
```

## Train the model
To train on Human3.6M with detected 2D poses by CPN as inputs, run:

```
python main.py --train --keypoints cpn_ft_h36m_dbb -n 'your_exp_name'
```
To train on Human3.6M with GT 2D poses as inputs, run:

```
python main.py --train --keypoints gt -n 'your_exp_name'
```
