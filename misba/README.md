# MISBA: Multiple-instance Inspired Shared-weight Bootrapped from Autoencoders model

This folder contains work developing a method to create a classifier from very little training data given that the features come from a mixture of examples from K kinds of units, and that the labels are the OR of all of the examples' labels. I am researching the effects of using "convolutional" models to share weights and reduce the parameters, and the effects of unsupervised training first to extract deeper features without overfitting.

I began with the construction of a purely synthetic dataset, and verified that naive approaches with off-the-wall models and one big feature space do not suffice. This is examined in `Constructing Synthetic Dataset.ipynb`. 

Then, I tested the ability of a simple model sharing weights between different instances to classify the synthetic dataset. Two forms of the model are evaluated and compared at different amounts of training data in `Training Classification Model.ipynb`.

## Future work:

The simple models with no pre-training perfectly learn the synthetic dataset. I may use another dataset which is more complex to construct a more challenging dataset deserving of pretraining, like collections of images. Then, I will evaluate the efficacy of a pretraining routine before input into the models already evaluated. Finally, misba.py will be filled out if the pretraining methods are successful.
