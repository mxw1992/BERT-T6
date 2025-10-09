# BERT-T6
![image](https://github.com/mxw1992/BERT-T6/blob/main/image/%E4%B8%BB%E5%9B%BE%E9%A2%9C%E8%89%B2.jpg)
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
