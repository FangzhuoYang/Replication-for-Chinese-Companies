import pandas as pd
import numpy as np
from scipy.stats import norm

# 读取 .csv 文件
df = pd.read_csv('/Users/yangfangzhuo/Desktop/助研/专利价值计算/cn_stock_patent.csv')

gamma = 0.007

def calculate_patent_value(df, gamma):
    """计算正态分布假设下的专利价值（与Stata代码完全一致）"""
    result_df = df.copy()
    
    try:
        # 步骤1: 缺失值处理（与Stata一致）
        result_df['ret_d0'] = result_df['ret_d0'].fillna(0)
        result_df['ret_d1'] = result_df['ret_d1'].fillna(0)
        result_df['ret_d2'] = result_df['ret_d2'].fillna(0)
       
        # 步骤2: 计算累计收益率 R（使用对数计算，与Stata一致）
        result_df['R'] = np.exp(np.log(1 + result_df['ret_d0']) + 
                               np.log(1 + result_df['ret_d1']) + 
                               np.log(1 + result_df['ret_d2'])) - 1
       
        # 步骤3: 计算波动率 v
        result_df['v'] = result_df['vol'] * np.sqrt(3)
     
        # 步骤4: 计算信噪比参数 delta
        result_df['delta'] = 1 - np.exp(-gamma)
        
        # 步骤5: 计算参数 a
        # 避免除以零的情况
        result_df['a'] = -np.sqrt(result_df['delta']) * result_df['R'] / result_df['v'].replace(0, np.nan)
        
        # 步骤6: 计算条件期望 m_graw3m0F（与Stata完全一致）
        a_values = result_df['a']
        pdf_values = norm.pdf(a_values)
        cdf_values = norm.cdf(a_values)
        
        # 避免除以零的情况
        safe_ratio = np.where(cdf_values < 1, pdf_values / (1 - cdf_values), 0)
        
        result_df['m_graw3m0F'] = (result_df['delta'] * result_df['R'] + 
                                  np.sqrt(result_df['delta']) * result_df['v'] * safe_ratio)
        
        # 步骤7: 转换为美元价值 mw_graw3m0F（与Stata一致，除以1000）
        result_df['mw_graw3m0F'] = result_df['m_graw3m0F'] * result_df['mkcap']
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        raise
    
    return result_df

def process_patent_values(df):
    """处理专利价值：平均分配并聚合数据"""
    
    # 首先计算每个公司每天的专利数量
    daily_patent_count = df.groupby(['Stkcd', 'date']).size().reset_index(name='patent_count')
    
    # 合并专利数量信息到原始数据
    df_with_count = pd.merge(df, daily_patent_count, on=['Stkcd', 'date'], how='left')
    
    # 将专利价值平均分配到每个专利
    df_with_count['mw_graw3m0F_avg'] = df_with_count['mw_graw3m0F'] / df_with_count['patent_count']
    
    # 检查哪些列存在，只聚合存在的列
    available_columns = []
    if 'year' in df_with_count.columns:
        available_columns.append(('year', 'first'))
    if 'DuplicateCount' in df_with_count.columns:
        available_columns.append(('DuplicateCount', 'first'))
    
    # 基础聚合列
    agg_dict = {
        'patent_count': 'first',
        'mw_graw3m0F_avg': 'first'
    }
    
    # 添加可选的列
    for col, agg_func in available_columns:
        agg_dict[col] = agg_func
    
    # 按公司日期聚合
    aggregated_df = df_with_count.groupby(['Stkcd', 'date']).agg(agg_dict).reset_index()
    
    return aggregated_df

def main(df):
    print("原始数据:")
    print(df.head())
    print(f"数据总行数: {len(df)}")
    print(f"数据列名: {df.columns.tolist()}")
    print("\n" + "=" * 80 + "\n")

    # 步骤1: 计算专利价值
    print("开始计算专利价值...")
    patent_value_df = calculate_patent_value(df, gamma)

    print("\n专利价值计算结果 (前10行):")
    available_columns = ['Stkcd', 'date', 'mkcap', 'R', 'v', 'delta', 'a', 'm_graw3m0F', 'mw_graw3m0F']
    available_columns = [col for col in available_columns if col in patent_value_df.columns]
    print(patent_value_df[available_columns].head(10))
    
    # 步骤2: 处理专利价值（平均分配并聚合）
    print("\n开始处理专利价值（平均分配并聚合）...")
    final_df = process_patent_values(patent_value_df)
    
    print("\n最终结果 (前10行):")
    print(final_df.head(10))
    print(f"\n最终数据行数: {len(final_df)}")
    print(f"最终数据列名: {final_df.columns.tolist()}")
    
    # 保存结果到文件
    output_file = '/Users/yangfangzhuo/Desktop/patent_value_results.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    return final_df

if __name__ == "__main__":
    final_df = main(df)