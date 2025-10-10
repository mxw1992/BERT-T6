# BERT-T6
By fine-tuning the ProtBert protein language model on T6SS effector (T6SE) classification tasks, BERT-T6 can learn directly from sequence data without relying on handcrafted features, achieving state-of-the-art prediction performance.
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
Parameters:
- ```--test_dir``` path to the input protein FASTA file.
- ```--result_dir``` directory that stores prediction outputs (CSV table).
- ```--checkpoint_path``` path to the model weights (download from [here](https://drive.google.com/drive/folders/1dPDHxY7ga4JVDC6R4OdPueeYdQjR_Peg)).

## Contact
Please contact Xianwei Mo at 13580342797@163.com for questions.
