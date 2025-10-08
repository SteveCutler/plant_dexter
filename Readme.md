### PLANT DEXTER v2

This is phase 2 of my deep learning model designed to be an offline nature guidebook
for backcountry campers or whoever else wants it.

The scope of this project will be to identify any Ontario native plant that could be encountered in the wild - using an on device, completely offline inference model.

This is a learning project so I will be using it to get a sense of the workflow involved in using a foundation model, creating a vector db, etc.

####

requirements:
-master list of plants native to ontario
-small training set (minimum 50 images per plant), likely sourced from inaturalist or other open source APIs
-a model to use

####

Doing more preliminary research I found Biotrove, and open source clip based model trained on a tonne of plant, insect, animal etc data. This could provide a good backbone to use
https://baskargroup.github.io/BioTrove/

They also provide a dataset of 40m images used in the training process, featuring a tonne of metadata including lat and long, taxonomies, etc. I'm going to use this as my source for image data. I might start with 100 images per species for 200 plant species to begin with to get the hang of the workflow and then build out from there.
https://huggingface.co/datasets/BGLab/BioTrove/viewer?row=58

####

I grabbed the All Species excel sheet from the Natural Heritage Information Center from the province of Ontario:

https://www.ontario.ca/page/get-natural-heritage-information?utm_source=chatgpt.com

It contains over 4000 vascular plants native to ontario as well as plants, animals, fungus, etc that come in handy later on

##

Initial commit: after narrowing down the ontario plant list to only vascular plants of somewhat common nature, I got ~3200 plants.
I streamed the BGLab/Biotrove dataset csv from hugging face, and crossreferencing through 12 million of the 168 million rows I came up with about 2400 unique plant overlaps for my dataset.

I copied these rows into a new csv with all the columns including common names, image urls etc, and then made a script to pull max 200 images per plant species onto an external drive. These should all be high quality research grade images.

Next step: working on embedding these vectors using the biotrove clip model and probably a fine tune train on the top layers. My goal is 95% top-5 accuracy with low on device latency.

##
