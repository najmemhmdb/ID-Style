# ID-Style: Identity-preserving Editing of Multiple Facial Attributes by Learning Global Edit Directions and Local Adjustments

<img src="https://github.com/najmemhmdb/ID-Style/blob/master/img/architecture.png" width="1000"/>

[[arXiv]](https://arxiv.org/pdf/2309.14267.pdf)[[code]](https://github.com/najmemhmdb/ID-Style) <br>
[Najmeh Mohammadbagheri](https://github.com/najmemhmdb)<sup>1</sup>, [Fardin Ayar](https://github.com/fardinayar)<sup>1</sup> <br>
<sup>1</sup>Amirkabir University of Technology, Iran. <br>

## Download pretrained models:

Download pretrained models from [here](https://drive.google.com/file/d/1ce6L4js-_mrh_uROmOY4Dbuoo81T8t8a/view?usp=drive_link).

## Preparing dataset

Follow the instruction in [here](https://github.com/yhlleo/stylegan-mmuit/tree/master/styleganv2) to generate data.

## Training 

```
$ bash train.sh
```

## Testing

```
$ bash test.sh
```

## Acknowledgement

This repo is built on several existing projects:

 - [StyleGAN2](https://github.com/NVlabs/stylegan)
 - [StyleFlow](https://github.com/RameenAbdal/StyleFlow)
 - [ISF-GAN](https://github.com/yhlleo/stylegan-mmuit)

### Citation

```
@InProceedings{mohammadbagheri2023identitypreserving,
    author    = {Mohammadbagheri, Najmeh and Ayar, Fardin and Nickabadi, Ahmad and Safabakhsh, Reza},
    title     = {Identity-preserving Editing of Multiple Facial Attributes by Learning Global Edit Directions and Local Adjustments},
    booktitle = {Computer vision and image understanding, 246, p.104047},
    year      = {2024}
}
```
