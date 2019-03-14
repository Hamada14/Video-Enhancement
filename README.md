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
 * [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
 * [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
 * [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
 * [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
 * [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
 * [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
 * [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]


 2. Run `flownet2-pytorch/install.bash` to install compiles the necessary libraries.


### Demo

The below code can be used to run inference using a model that's stored locally and will produce a `.flo` file for each two consecutive pictures in the dataset folder.

<span style="color:blue">Remember to modify `EXAMPLES_DIR` and `MODEL_PATH` to the directories on your machine.</span>
```bash
$ export EXAMPLES_DIR=/home/hamada14/Video-Enhancement/flownet2-pytorch/examples
$ export MODEL_PATH=/home/hamada14/Video-Enhancement/flownet2-pytorch/models/FlowNet2-S_checkpoint.pth.tar
$ python3 flownet2-pytorch/main.py --inference --model FlowNet2S --save_flow \
    --save flownet2-pytorch/examples/ --inference_dataset ImagesFromFolder \
    --inference_dataset_root $EXAMPLES_DIR --resume $MODEL_PATH --skip_training \
     --inference_dataset_iext ppm
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
