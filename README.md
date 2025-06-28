# ct-lrp
Code for Contrastive Token LRP

Run the training code with the following line:

```python train_model_working.py --datasetname Twitter --modelname BiGCN```

Train on either Twitter/PHEME/Weibo datasets.

On certain systems, ```argparse``` may throw an error for the ```--modelname``` argument. If this occurs, manually edit the variable in the file to select the desired model.

To perform the evaluation with the desired model and explanation method, run the evaluation code:

```python run_expl_eval_batched.py --modelname BiGCN --exp_method ct-lrp```

The script will run the evaluation on all three datasets. As with the training file, manually edit the relevant variables if argparse throws an error.
