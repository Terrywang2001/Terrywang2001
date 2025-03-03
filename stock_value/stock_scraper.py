from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions  # 改用Edge的Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime
import threading
from typing import Dict, Optional
import os
import pandas as pd

app = FastAPI()
class StockIndexScraper:
    def __init__(self):
        self.szse_url = "https://www.szse.cn/market/index.html"
        self.sse_url = "https://www.sse.com.cn/"
        self.csv_file = 'stock_indices_history.csv'
        self.max_file_size = 5 * 1024 * 1024  # 5MB

        # 设置edge选项
        self.edge_options = EdgeOptions()
        self.edge_options.add_argument('--headless')
        self.edge_options.add_argument('--disable-gpu')
        self.edge_options.add_argument('--no-sandbox')
        self.edge_options.add_argument('--disable-dev-shm-usage')
        self.edge_options.add_argument('--disable-extensions')
        self.edge_options.add_argument('--disable-software-rasterizer')
        self.edge_options.add_argument('--ignore-certificate-errors')
        self.edge_options.add_argument('--ignore-ssl-errors')

    def setup_driver(self):
        """
        初始化并配置WebDriver
        """
        try:
            # 创建edge浏览器实例
            self.driver = webdriver.Edge(options=self.edge_options)
            self.driver.implicitly_wait(5)  # 设置隐式等待时间
            return True
        except Exception as e:
            print(f"设置WebDriver时发生错误: {e}")
            return False

    def fetch_szse_data(self):
        """
        获取页面数据并提取目标信息
        """
        try:
            print("正在获取深证指数...")
            self.driver.get(self.szse_url)
            
            # 等待页面加载完成
            #time.sleep(0.1)  # 确保JavaScript有足够时间执行
            
            # 使用显式等待来确保元素加载完成
            wait = WebDriverWait(self.driver, 1)
            
            # 尝试多个可能的选择器
            selectors = [
                ".newest"
            ]
            selector = selectors[0]
            element = None
            element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
            
            # 获取日期
            try:
                print("\n等待深证日期数据加载...")
                
                # 使用更精确的选择器
                date_selector = "span.update-time"
                
                def date_loaded(driver):
                    elements = driver.find_elements(By.CSS_SELECTOR, date_selector)
                    return len(elements) >= 2 and elements[1].get_attribute('innerHTML').strip() != ""
                
                wait.until(date_loaded)
                
                date_elements = self.driver.find_elements(By.CSS_SELECTOR, date_selector)
                full_date_text = date_elements[1].get_attribute('innerHTML').strip() 
                
                # 截取日期部分 YYYY-MM-DD
                if full_date_text:
                    date_text = full_date_text[:10]  # 只取前10个字符，即YYYY-MM-DD部分
                    print(f"\n获取到的深证日期: {date_text}")
                else:
                    date_text = None
                    print("\n未获取到有效日期")
                
            except Exception as date_error:
                print(f"\n获取深证日期时发生错误: {date_error}")
                date_text = None

            if element:
                # 获取元素文本和类名
                value = element.text.strip()
                
                return {
                    'index_name': '深证指数',
                    'value': value,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'date': date_text
                }
            else:
                print("未找到目标元素")
                return None
                
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            return None
        
    def fetch_sse_data(self):
        """获取上证指数数据"""
        try:
            print("\n正在获取上证指数数据...")
            self.driver.get(self.sse_url)
            time.sleep(0.1)  # 等待页面加载
            wait = WebDriverWait(self.driver, 3)


            # 修复日期元素获取 
            try:
                # 定义日期等待条件
                def date_loaded(driver):
                    elements = driver.find_elements(By.CSS_SELECTOR, 
                        "div.sse_home_row.gray_bg span.new_date")
                    if elements:
                        value = elements[0].get_attribute('innerHTML').strip()
                        return value != "" and value != "-"
                    return False
                # 等待日期加载
                wait.until(date_loaded)
                # 获取加载后的日期
                date_element = self.driver.find_element(By.CSS_SELECTOR, 
                    "div.sse_home_row.gray_bg span.new_date")
                full_date_text = date_element.get_attribute('innerHTML').strip()
                
                # 新的日期格式化逻辑
                if full_date_text:
                    # 直接截取前10个字符，格式为：YYYY-MM-DD
                    date_text = full_date_text[5:15]
                    print(f"\n获取到的日期: {date_text}")
                else:
                    date_text = None
                    print("\n未获取到有效日期")
            except Exception as date_error:
                print(f"\n获取日期时发生错误: {date_error}")
                date_text = None

            # 等待并获取上证指数值
            print("\n等待指数数据加载...")
            # 定义等待条件：等待直到第三个元素的值不为 "-"
            def value_loaded(driver):
                elements = driver.find_elements(By.CSS_SELECTOR, ".hq_index.szzs i")
                if len(elements) >= 3:
                    value = elements[2].get_attribute('innerHTML').strip()
                    return value != "-" and value != ""
                return False
            
            # 等待数据加载
            wait.until(value_loaded)
                
            # 获取所有i标签
            elements = self.driver.find_elements(By.CSS_SELECTOR, ".hq_index.szzs i")
            
            # 从HTML中提取数值
            if len(elements) >= 3:
                target_element = elements[2]
                index_value = target_element.get_attribute('innerHTML').strip()
                
            if index_value:
                value = index_value
                print(f"\n获取到的指数值: {value}")
                    
                return {
                        'index_name': '上证指数',
                        'value': value,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'date': date_text
                    }
            else:
                print("\n未找到指数值元素")
            # if row:
            #     # 获取所有列数据
            #     columns = row.find_elements(By.TAG_NAME, "td")
            #     if len(columns) >= 7:  # 确保有足够的列
            #         latest_price = columns[6].text.strip()
                    
            #         return {
            #             'index_name': '上证指数',
            #             'value': latest_price,
            #             'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            #             'date': date_text
            #         }
        except Exception as index_error:
            print(f"\n获取指数值时发生错误: {index_error}")
            
            # 输出当前页面状态
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, ".hq_index.szzs i")
                print("\n当前找到的元素数量:", len(elements))
                for i, elem in enumerate(elements):
                    print(f"元素 {i+1} 的文本:", elem.text)
                    print(f"元素 {i+1} 的HTML:", elem.get_attribute('outerHTML'))
            except:
                pass

            
        return None

        
    def save_to_csv(self, data):
        """保存数据到CSV文件"""
        try:
            new_data = pd.DataFrame(data)
            if os.path.exists(self.csv_file):
                existing_data = pd.read_csv(self.csv_file)
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                updated_data = new_data
            
            updated_data.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
            self.manage_csv_size()
            print(f"\n数据已保存到 {self.csv_file}")
        except Exception as e:
            print(f"保存数据时发生错误: {e}")

    def manage_csv_size(self):
        """管理CSV文件大小"""
        if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > self.max_file_size:
            df = pd.read_csv(self.csv_file)
            # 保留后半部分的数据
            df = df.iloc[len(df)//2:]
            df.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
        
    def run(self):
        """运行爬虫并获取两个指数的数据"""
        print("开始设置浏览器...")
        if not self.setup_driver():
            print("浏览器设置失败")
            return
            
        try:
            # 获取两个指数的数据
            szse_data = self.fetch_szse_data()
            sse_data = self.fetch_sse_data()
            
            # 整理数据用于展示和保存
            all_data = []
            result = {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
            if szse_data:
                # print("\n深证成指数据:")
                # for key, value in szse_data.items():
                #     print(f"{key}: {value}")
                all_data.append(szse_data)
                result["深证成指"] = {
                    "当前值": szse_data['value'],
                    "日期": szse_data['date']
                }
                
            if sse_data:
                # print("\n上证指数数据:")
                # for key, value in sse_data.items():
                #     print(f"{key}: {value}")
                all_data.append(sse_data)
                result["上证指数"] = {
                    "当前值": sse_data['value'],
                    "日期": sse_data['date']
                }
            
            # 保存数据到CSV
            if all_data:
                self.save_to_csv(all_data)
                return result
            else:
                print("\n未能获取任何数据")
                return None
                
        except Exception as e:
            print(f"运行时发生错误: {e}")
            return None
        finally:
            print("\n清理资源...")
            self.driver.quit()

@app.get("/get_indices")
async def get_indices():
    """获取深证指数和上证指数"""
    scraper = StockIndexScraper()
    result = scraper.run()
    if not result:
        raise HTTPException(status_code=500, detail="获取数据失败")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)