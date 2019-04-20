# Video-Enhancement

## Directories Walkthrough
* **flownet2-pytorch**

 Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925).

* **pytorch-sepconv**

 Pytorch implementation of [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/abs/1708.01692)

## FlowNet2
### Installation

1. Download the FlowNet2 pretrained model to `flownet2-pytorch/models`

 The available models are available below:
 * [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB] ** (Please Download this model and avoid changing the name) **
 * [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
 * [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
 * [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
 * [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
 * [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
 * [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]


 2. Move the downloaded model to `flownet2/models` directory while keeping its name constant as `FlowNet2_checkpoint.pth.tar`.

 3. Run `flownet2-pytorch/install.bash` to compile the necessary libraries.

 4. Install the necessary libraries using pip:
  ```bash
  $ pip3 install tensorboardX setproctitle colorama tqdm scipy pytz cvbase opencv-python
  ```

### Demo

The below code can be used to run inference using a model that's stored locally and will produce a `.flo` file for each two consecutive pictures in the dataset folder.

There are two demo functions in `flow_model_wrapper.py` that can be used.
Each of the functions demonstrates either calculating the flow for a specific directory or for a pair of images.
```bash
$ python3 flow_model_wrapper.py
```

### Visualizing Optical Flow
* To visualize a flow file:
```python
# You might need to run `pip install cvbase` first
import cvbase as cvb
# to visualize a flow file
cvb.show_flow('result.flo')
```

* To create a random flow and visualize it:
```python
# run `pip install cvbase` first
import cvbase as cvb
# to visualize a loaded flow map
flow = np.random.rand(100, 100, 2).astype(np.float32)
cvb.show_flow(flow)
```

## FRVSR

### Trainer
In order for the trainer to work correctly a dataset must exist in a directory called `data_set`. The directory should contain any videos that should be used in the training process.

To start the training process, run the following command:
`python frvsr_trainer.py`
