# BERT-T6
A BERT-based transfer learning approach for T6SE prediction
## Set up
## Usage
### Train model
### Prediction
You can predict your interested ptotein,  here we give an example of T6SE prediction:
```
cd ./PredictT6SE
python predictT6SE.py --test_dir ./hypo.fasta --result_dir ./hypo.csv --checkpoint_path ./best_model_train_valid_set.pt
```
## Contact
Please contact Xianwei Mo at 13580342797@163.com for questions.
