import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv('/Users/yangfangzhuo/Desktop/Calculation/cn_stock_patent.csv')

gamma = 0.007

def calculate_patent_value(df, gamma):
    result_df = df.copy()
    
    try:
        result_df['ret_d0'] = result_df['ret_d0'].fillna(0)
        result_df['ret_d1'] = result_df['ret_d1'].fillna(0)
        result_df['ret_d2'] = result_df['ret_d2'].fillna(0)
       
        result_df['R'] = np.exp(np.log(1 + result_df['ret_d0']) + 
                               np.log(1 + result_df['ret_d1']) + 
                               np.log(1 + result_df['ret_d2'])) - 1
       
        result_df['v'] = result_df['vol'] * np.sqrt(3)
     
        result_df['delta'] = 1 - np.exp(-gamma)
        
        result_df['a'] = -np.sqrt(result_df['delta']) * result_df['R'] / result_df['v'].replace(0, np.nan)
        
        a_values = result_df['a']
        pdf_values = norm.pdf(a_values)
        cdf_values = norm.cdf(a_values)
        
        safe_ratio = np.where(cdf_values < 1, pdf_values / (1 - cdf_values), 0)
        
        result_df['m_graw3m0F'] = (result_df['delta'] * result_df['R'] + 
                                  np.sqrt(result_df['delta']) * result_df['v'] * safe_ratio)
        
        result_df['mw_graw3m0F'] = result_df['m_graw3m0F'] * result_df['mkcap']
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        raise
    
    return result_df

def process_patent_values(df):
    
    daily_patent_count = df.groupby(['Stkcd', 'date']).size().reset_index(name='patent_count')
    
    df_with_count = pd.merge(df, daily_patent_count, on=['Stkcd', 'date'], how='left')
    
    # 对 R 和 m_graw3m0F 也进行平均分配
    df_with_count['mw_graw3m0F_avg'] = df_with_count['mw_graw3m0F'] / df_with_count['patent_count']
    df_with_count['R_avg'] = df_with_count['R'] / df_with_count['patent_count']
    df_with_count['m_graw3m0F_avg'] = df_with_count['m_graw3m0F'] / df_with_count['patent_count']
    
    available_columns = []
    if 'year' in df_with_count.columns:
        available_columns.append(('year', 'first'))
    if 'DuplicateCount' in df_with_count.columns:
        available_columns.append(('DuplicateCount', 'first'))
    
    agg_dict = {
        'patent_count': 'first',
        'mw_graw3m0F_avg': 'first',
        'R_avg': 'first',
        'm_graw3m0F_avg': 'first'
    }
    
    # 添加原始值的总和（可选，根据你的需求选择）
    agg_dict.update({
        'R': 'sum',  # 或者 'mean' 根据你的需求
        'm_graw3m0F': 'sum',  # 或者 'mean' 根据你的需求
        'mw_graw3m0F': 'sum'  # 保留总价值
    })
    
    for col, agg_func in available_columns:
        agg_dict[col] = agg_func
    
    aggregated_df = df_with_count.groupby(['Stkcd', 'date']).agg(agg_dict).reset_index()
    
    return aggregated_df

def main(df):
    print("原始数据:")
    print(df.head())
    print(f"数据总行数: {len(df)}")
    print(f"数据列名: {df.columns.tolist()}")
    print("\n" + "=" * 80 + "\n")

    print("开始计算专利价值...")
    patent_value_df = calculate_patent_value(df, gamma)

    print("\n专利价值计算结果 (前10行):")
    available_columns = ['Stkcd', 'date', 'mkcap', 'R', 'v', 'delta', 'a', 'm_graw3m0F', 'mw_graw3m0F']
    available_columns = [col for col in available_columns if col in patent_value_df.columns]
    print(patent_value_df[available_columns].head(10))
    
    print("\n开始处理专利价值（平均分配并聚合）...")
    final_df = process_patent_values(patent_value_df)
    
    print("\n最终结果 (前10行):")
    print(final_df.head(10))
    print(f"\n最终数据行数: {len(final_df)}")
    print(f"最终数据列名: {final_df.columns.tolist()}")
    
    # 显示统计信息
    print("\n数据统计信息:")
    print(final_df.describe())

    output_file = '/Users/yangfangzhuo/Desktop/Calculation/patent_value_results_extended.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    return final_df

if __name__ == "__main__":
    final_df = main(df)
