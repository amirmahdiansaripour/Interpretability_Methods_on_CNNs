# Interpretability_Methods_on_CNNs

In this project, I implemented and explained three interpretability methods on Convolutional Neural Networks: **Class Activation Map (CAM)**, **Gradient-Weighted CAM (Grad-CAM)**, and **Saliency Map**. 
These methods help identify the regions (features) of an input image that have the most impact on a CNN's predictions. 

The famous **VGG16 network** and cats_vs_dogs dataset are used in this project. Since the network was originally trained on the ImageNet dataset, I applied transfer learning and modified the network's top layers to adapt it to the cats_vs_dogs dataset.

1. **Grad-CAM** **([paper link](https://arxiv.org/pdf/1610.02391))**: This method assigns a weight to each feature (filter) of the activation map. The weights are calculated based on the following formula:

 $$\alpha^c = \frac{1}{Z} \Sigma_i \Sigma_j \frac{\partial y^c}{\partial A_{ij}}$$

 Where $\alpha^c$ is the weights vector for the activation map $A$, which has $i * j$ filters (features). $Z$ is a normalization factor ($Z = i * j$).  
 When the weights are calculated, they are dotted with their corresponding features:

 $$L^c_{Grad-CAM} = ReLU(\alpha^c \; A^k)$$

$L^c_{Grad-CAM}$ above is a heatmap showing important regions of the image in classification. We use $ReLU$ because negative values are not defined for pixels.

![img00](./images/GradCAM_output.png)
