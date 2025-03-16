# Enhancing Fine-Grained Image Classification with Adaptive Pseudo-Labeling

This repo contains the code for our project for the CS682 - Neural Networks code. We enhanced the implementation from the below paper for our specific use case:

- **A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification**, Jong-Chyi Su, Zezhou Cheng, and Subhransu Maji, CVPR 2021. [[paper](https://arxiv.org/abs/2104.00679), [poster](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/poster.pdf), [slides](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/slides.pdf)]
- **Semi-Supervised Learning with Taxonomic Labels**, Jong-Chyi Su and Subhransu Maji, BMVC 2021. [[paper](https://arxiv.org/abs/2111.11595), [slides](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/slides_bmvc.pdf)]


## Fine-Grained Image Classification
Fine-grained image classification represents the task of distinguishing between visually similar classes, such as species of birds, or species of fungi. These tasks suffer from long-tailed class distributions where some classes have abundant labeled data while others have very few samples. This imbalance makes it challenging for models to learn effectively across all classes.

## Fixed Threshold Pseudo-labeling
Given the scarcity of annotated data, semi-supervised learning (SSL) has emerged more promising than supervised approaches. SSL approaches utilize a small set of labeled images to guide the classification of a much larger unlabeled dataset. One widely used SSL method is pseudo-labeling. How does Pseudo-labeling work?
A classifier is initially trained on the small amount of labeled data.
2. The trained model then predicts labels on the unlabeled data, thus creating pseudo-labels
3. The most confident pseudo-labels are added to the labeled dataset, and the model is iteratively retrained.
Selecting confident pseudo-labels is crucial, and the most common approach is to use a fixed threshold (τ), where pseudo-labels with a probability greater than τ are considered reliable. However, this method has notable drawbacks:
As the model learns, its confidence in predictions changes. A fixed threshold does not adapt to this evolution, making the learning process suboptimal.
Different classes vary in their characteristics, representation, and difficulty. A single threshold may disproportionately favor overrepresented classes while ignoring underrepresented ones.

## Adaptive Thresholding
Fixed-threshold pseudo-labeling can introduce noisy labels if the threshold is too low or exclude valuable data if the threshold is too high. This motivates our approach of using an adaptive thresholding strategy, which:
- Dynamically decays over training iterations to match the model’s evolving confidence. A higher threshold ensures that only the most confident predictions are used early in training, reducing incorrect pseudo-label propagation. Over time, as the model becomes more reliable, the threshold is reduced to allow more predictions to contribute to training.
- Incorporates class-specific thresholds to better handle class imbalances and ensure a more balanced learning process. Class-specific thresholds enable the model to handle each class separately and adapt the learning process based on model confidence and class representation.
By implementing an adaptive pseudo-labeling approach, we aim to improve fine-grained classification performance, especially in datasets with long-tailed distributions.

## Datasets
- **Semi-Fungi**: dataset build from the [2018 FGVCx Fungi Classification Challenge](https://github.com/visipedia/fgvcx_fungi_comp) at [FGVC5 workshop](https://sites.google.com/view/fgvc5) at CVPR 2018.
  

The splits of each of these datasets can be found under
```data/${dataset}/${split}.txt``` corresponding to:
- l_train -- labeled in-domain data
- u_train_in -- unlabeled in-domain data
- u_train_out -- unlabeled out-of-domain data
- u_train (combines u_train_in and u_train_out)
- val -- validation set
- l_train_val (combines l_train and val)
- test -- test set

## Experiment Setup

Our experiments were conducted on the Semi-Fungi dataset, a long-tailed dataset with 200 classes. We use RESNET-18 as the base model.

## Results
- **Adaptive thresholding mitigates pseudo-label skewness:** Our algorithm produced a more balanced distribution of pseudo-labels compared to the static method.

- **Change in accuracy:** Underrepresented classes show improved accuracy. Overrepresented classes experience a slight drop in accuracy.

- **Overall accuracy increases:** We also show that by improving the performance on underrepresented classes, we are able to produce a small boost in overall accuracy compared to the static method


## Training

To train the model, use the following command:
```
CUDA_VISIBLE_DEVICES=0 python run_train.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --trainval
```


## Citation 
```
@inproceedings{su2021realistic,
  author    = {Jong{-}Chyi Su and Zezhou Cheng and Subhransu Maji},
  title     = {A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}

@inproceedings{su2021taxonomic,
  author    = {Jong{-}Chyi Su and Subhransu Maji},
  title     = {Semi-Supervised Learning with Taxonomic Labels},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2021}
}

@article{su2021semi_iNat,
      title={The Semi-Supervised iNaturalist Challenge at the FGVC8 Workshop}, 
      author={Jong-Chyi Su and Subhransu Maji},
      year={2021},
      journal={arXiv preprint arXiv:2106.01364}
}

@article{su2021semi_aves,
      title={The Semi-Supervised iNaturalist-Aves Challenge at FGVC7 Workshop}, 
      author={Jong-Chyi Su and Subhransu Maji},
      year={2021},
      journal={arXiv preprint arXiv:2103.06937}
}
```
