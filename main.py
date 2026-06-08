import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, snapshot_download
import sys
import akshare as ak
import pandas as pd
from datetime import datetime

# --- 配置部分 ---
MODEL_ID = "Qwen/Qwen3.5-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 正在初始化 Qwen3.5-2B 模型...")
print(f"💡 运行设备: {DEVICE.upper()}")

try:
    # 1. 下载/获取模型本地路径
    print("⏳ 检查模型文件 (首次运行需下载)...")
    model_path = snapshot_download(MODEL_ID)
    print(f"✅ 模型路径: {model_path}")

    # 2. 加载分词器
    print("⏳ 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision="master"
    )

    # 3. 加载模型权重
    print(f"⏳ 加载模型到 {DEVICE.upper()} ...")
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            revision="master"
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            revision="master"
        ).eval()
    print("✅ 模型加载完成！ ready to chat.\n")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    print("💡 请检查网络连接或依赖库 (pip install modelscope torch transformers akshare pandas)")
    sys.exit(1)


# --- 金融数据工具函数 ---
def get_stock_realtime(symbol):
    """
    功能一：获取A股实时行情
    :param symbol: 股票名称或代码 (如 '贵州茅台' 或 '600519')
    :return: 格式化的字符串结果
    """
    try:
        # 获取全市场实时数据
        df = ak.stock_zh_a_spot_em()

        # 筛选目标股票 (支持通过名称或代码搜索)
        target_df = df[(df['代码'] == symbol) | (df['名称'] == symbol)]

        if target_df.empty:
            return f"未找到股票 '{symbol}' 的数据。请检查名称或代码是否正确。"

        # 提取关键数据
        name = target_df.iloc[0]['名称']
        code = target_df.iloc[0]['代码']
        price = target_df.iloc[0]['最新价']
        change_pct = target_df.iloc[0]['涨跌幅']
        volume = target_df.iloc[0]['成交量']
        amount = target_df.iloc[0]['成交额']

        return f"""
✅ **实时行情查询结果**

- **股票名称**: {name} ({code})
- **当前价格**: **{price}** 元
- **涨跌幅**: {change_pct}%
- **成交量**: {volume} 手
- **成交额**: {amount} 元

*(数据来源: 东方财富)*
"""
    except Exception as e:
        return f"获取实时数据失败: {str(e)}"


def get_stock_history(symbol, start_date="20230101", end_date=None):
    """
    功能二：获取历史行情并分析
    :param symbol: 股票代码 (A股通常为6位数字)
    :param start_date: 开始日期 (格式: 20230101)
    :param end_date: 结束日期 (格式: 20231231)
    :return: 分析结果
    """
    try:
        # 如果未指定结束日期，默认为今天
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 获取历史数据
        # 注意：AkShare 接口通常需要代码，如果用户输入名称，需要先转换
        # 这里简化处理，假设用户输入代码，或者你需要写一个名称转代码的逻辑
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")

        if df.empty:
            return f"未找到股票代码 '{symbol}' 在指定时间段的历史数据。"

        # 数据分析
        # 将日期列设置为索引以便分析
        df.set_index('日期', inplace=True)
        # 将字符串价格转为浮点数
        df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
        df['最高'] = pd.to_numeric(df['最高'], errors='coerce')
        df['最低'] = pd.to_numeric(df['最低'], errors='coerce')

        # 计算指标
        highest_price = df['最高'].max()
        lowest_price = df['最低'].min()
        avg_close = df['收盘'].mean()

        return f"""
📊 **历史行情分析报告**

- **分析标的**: 股票代码 {symbol}
- **时间范围**: {start_date} 至 {end_date}
- **期间最高价**: {highest_price}
- **期间最低价**: {lowest_price}
- **平均收盘价**: {avg_close:.2f}

*(注: 本分析基于日K线数据)*
"""
    except Exception as e:
        return f"获取历史数据失败: {str(e)}"


def get_macro_data(indicator):
    """
    功能三：获取宏观经济数据 (修复版)
    """
    try:
        if indicator.lower() in ['cpi', '消费者价格指数']:
            df = ak.macro_china_cpi_yearly()
            target_col_keyword = "CPI"  # 用于模糊匹配列名

        elif indicator.lower() in ['gdp', '国内生产总值']:
            # 修正拼写错误：macro_china_gdp_yearly
            df = ak.macro_china_gdp_yearly()
            target_col_keyword = "GDP"  # 用于模糊匹配列名

        else:
            return "暂不支持该宏观经济指标。"

        if df.empty:
            return "暂无相关数据。"

        # --- 动态获取列名逻辑 (防止AkShare更新导致报错) ---
        latest = df.iloc[-1]
        columns = df.columns.tolist()

        # 1. 获取时间列 (通常是第一列，或者包含'时间'/'年份'的列)
        time_col = columns[0]
        for col in columns:
            if '时间' in col or '年份' in col or '年' == col:
                time_col = col
                break

        # 2. 获取数值列 (包含关键词的列，或者是第二列)
        value_col = columns[1] if len(columns) > 1 else columns[0]
        for col in columns:
            if target_col_keyword in col or '值' in col or '绝对值' in col:
                value_col = col
                break

        year = latest[time_col]
        value = latest[value_col]

        title = "📈 中国CPI数据" if 'cpi' in indicator.lower() else "💰 中国GDP数据"
        unit = "%" if 'cpi' in indicator.lower() else "亿元"

        return f"{title}\n- **统计年份**: {year}\n- **数值**: {value} {unit}\n*(数据来源: 国家统计局)*"

    except Exception as e:
        return f"获取宏观数据失败: {str(e)}"


def chat(user_input, history=None):
    """
    增强版聊天函数：包含意图识别和工具调用
    """
    user_input_lower = user_input.lower()

    try:
        # ==========================================
        # 🔥 优先级 1：宏观经济数据 (提到最前面)
        # ==========================================
        if any(keyword in user_input_lower for keyword in ['cpi', 'gdp', '国内生产总值', '消费者价格指数', '宏观数据']):
            if 'gdp' in user_input_lower or '国内生产总值' in user_input_lower:
                return get_macro_data('gdp')
            elif 'cpi' in user_input_lower or '消费者价格指数' in user_input_lower:
                return get_macro_data('cpi')
            else:
                # 如果只检测到“宏观”但没指定具体指标
                return "您想查询 CPI 还是 GDP？请具体说明。"

        # ==========================================
        # 🔥 优先级 2：股票实时行情
        # ==========================================
        elif any(keyword in user_input_lower for keyword in ['股价', '行情', '最新价', '实时']):
            # 简单的关键词提取逻辑
            if '茅台' in user_input:
                return get_stock_realtime('贵州茅台')
            elif '平安' in user_input:
                return get_stock_realtime('平安银行')
            else:
                # 尝试提取6位数字代码
                import re
                code_match = re.search(r'\d{6}', user_input)
                if code_match:
                    return get_stock_realtime(code_match.group())
                else:
                    return "请告诉我具体的股票名称（如茅台）或代码（如600519）。"

        # ==========================================
        # 🔥 优先级 3：历史走势分析
        # ==========================================
        elif any(keyword in user_input_lower for keyword in ['历史', 'k线', '走势', '分析']):
             # 简单逻辑，实际可根据需要增强
             import re
             code_match = re.search(r'\d{6}', user_input)
             if code_match:
                 return get_stock_history(code_match.group())
             else:
                 return "历史分析功能请提供股票代码（如：分析 600519 的走势）。"

        # ==========================================
        # 🔥 优先级 4：通用对话 (调用大模型)
        # ==========================================
        else:
            messages = [
                {"role": "user", "content": user_input}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

    except Exception as e:
        return f"❌ 执行出错: {str(e)}"


if __name__ == "__main__":
    print("🤖 金融数据分析助手已就绪 (输入 'exit' 退出)\n")
    print("💡 支持指令：查询实时股价、分析历史K线、查询CPI/GDP等\n")

    while True:
        try:
            user_input = input("👤 用户: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break

            print("🤖 助手: ", end="", flush=True)
            response = chat(user_input)
            print(response)
            print()  # 空行分隔

        except KeyboardInterrupt:
            print("\n\n👋 强制退出。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")