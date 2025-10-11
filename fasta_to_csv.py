import pandas as pd
import argparse
import os

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Convert FASTA file to CSV format')
    parser.add_argument('input_file', help='Input FASTA file path')
    parser.add_argument('output_file', help='Output CSV file path')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    try:
        with open(args.input_file, 'r') as f:
            content = f.read()
            seq = content.split('>')
        del(seq[0])

        df = pd.DataFrame(columns=['ID', 'SEQUENCE', 'SEQUENCE_space', 'Label'])
        
        for i in range(len(seq)):
            a = seq[i].split('|')
            df.loc[i] = [a[0], a[2].strip(), " ".join(a[2].strip()), a[1]]
        
        df.to_csv(args.output_file, index=False)
        print(f"Successfully converted {args.input_file} to {args.output_file}")
        
    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    main()
