<h2>Implementation of Character Region Awareness of Text Detection (CRAFT) Paper</h2>

<h3>Model Architecture Files :</h3>

<b>basenet/vgg16.py</b>
1) Downloads pre-trained VGG-16 with Batch Normalization network from PyTorch's torchvision library.
2) Slices this network into 5 stages : Stage 1-5 - These slices become useful when establishing skip-connections with the UNet network defined in craft.py
3) Also, creates Stage 6 block (as described in the paper). This block also becomes useful when creating skip-connection with the Unet network.
4) Returns the output of this VGG-16 network (stage 1-5) and stage 6 from the <i>forward</i> method.

<b>craft.py</b>
1) Uses the network defined in basenet/vgg16.py as the base network.
2) Creates Unet and establishes skip connection between the base network and Unet network.
3) Gives Region score and Affinity score as output.

To run <b>craft.py</b>, clone the repository and run the following command :
<b>python craft.py</b>
