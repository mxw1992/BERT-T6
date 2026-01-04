# -*- coding: utf-8 -*-
"""
Created on Thu May 22 21:58:36 2025

@author: Lenovo
"""
def merge_fasta_files(file1_path, file2_path, output_path):
    """
    合并两个FASTA文件到一个输出文件
    
    参数:
        file1_path (str): 第一个FASTA文件路径
        file2_path (str): 第二个FASTA文件路径
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w') as out_file:
        # 写入第一个文件内容
        with open(file1_path, 'r') as f1:
            out_file.write(f1.read())
        
        # 写入第二个文件内容
        with open(file2_path, 'r') as f2:
            out_file.write(f2.read())

# 使用示例
file1 = "./neg_samples_train_set_0.fasta"  # 第一个FASTA文件路径
file2 = "./pos_samples_train_set_0.fasta"  # 第二个FASTA文件路径
output = "./samples_train_set_0.fasta" # 输出文件路径

merge_fasta_files(file1, file2, output)
print(f"FASTA文件已合并到 {output}")