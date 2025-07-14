# foundry_engine/data_generator.py (V3 - 动态验证版)

import os
import asyncio
import json
import time
import itertools
from typing import List, Dict, Any, Callable, Type, Set

import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {}
key_cycler = None
API_CALL_TIMEOUT_SECONDS = 90

def _initialize_api_keys():
    global API_KEYS, key_cycler
    if API_KEYS: return
    api_keys_str = os.getenv("GEMINI_API_KEYS")
    if not api_keys_str: raise ValueError("错误：未找到 GEMINI_API_KEYS 环境变量。")
    API_KEYS = {key.strip(): {'status': 'ready', 'unlock_time': 0} for key in api_keys_str.split(',') if key.strip()}
    if not API_KEYS: raise ValueError("错误：GEMINI_API_KEYS 变量为空或格式不正确。")
    print(f"成功从 .env 文件加载 {len(API_KEYS)} 个 API Key。")
    key_cycler = itertools.cycle(API_KEYS)

async def _fetch_batch(key: str, prompt: str, model_name: str, expected_keys: Set[str]) -> List[Dict] | None:
    """使用指定的Key和模型执行一次API请求，并根据期望的键进行验证。"""
    try:
        genai.configure(api_key=key)
        generation_config = genai.GenerationConfig(temperature=1.0, top_p=0.95, top_k=64, response_mime_type="application/json")
        model = genai.GenerativeModel(model_name, generation_config=generation_config)
        
        print(f"\n[调试] Worker 使用 Key ...{key[-4:]} 发起API请求...")
        response = await asyncio.wait_for(model.generate_content_async(prompt), timeout=API_CALL_TIMEOUT_SECONDS)
        
        data = json.loads(response.text)
        if not isinstance(data, list):
            print(f"\n[警告] Key ...{key[-4:]} 返回了非列表数据。")
            return None
        
        def is_valid_item(item):
            return isinstance(item, dict) and expected_keys.issubset(item.keys())

        return [item for item in data if is_valid_item(item)]

    except asyncio.TimeoutError:
        print(f"\n[警告] Key ...{key[-4:]} API请求超时 ({API_CALL_TIMEOUT_SECONDS}秒)。将进行冷却和重试。")
        raise
    except Exception as e:
        raise e

async def _worker(prompt: str, output_file: str, pbar: tqdm, model_name: str, expected_keys: Set[str]):
    """一个工作单元，负责不断请求直到成功获取一个批次的数据。"""
    while True:
        key_to_use = None
        while key_to_use is None:
            for key, state in API_KEYS.items():
                if state['status'] == 'ready':
                    key_to_use = key
                    break
            if key_to_use is None:
                await asyncio.sleep(1)
                for key, state in API_KEYS.items():
                    if state['status'] == 'cooldown' and time.time() > state['unlock_time']:
                        API_KEYS[key]['status'] = 'ready'
        try:
            API_KEYS[key_to_use]['status'] = 'in_use'
            batch_data = await _fetch_batch(key_to_use, prompt, model_name, expected_keys)
            if batch_data:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in batch_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                pbar.update(len(batch_data))
                API_KEYS[key_to_use]['status'] = 'ready'
                print(f"\n[成功] Key ...{key_to_use[-4:]} 成功获取一批数据 ({len(batch_data)}条)，已更新进度。")
                return
            else:
                print(f"\n[警告] Key ...{key_to_use[-4:]} 返回的数据经验证后为空。将冷却5秒后重试。")
                API_KEYS[key_to_use]['status'] = 'cooldown'
                API_KEYS[key_to_use]['unlock_time'] = time.time() + 5

        except Exception as e:
            key_display = key_to_use[-4:]
            if "429" in str(e) and "quota" in str(e).lower():
                cooldown_seconds = 61
                unlock_time = time.time() + cooldown_seconds
                API_KEYS[key_to_use]['status'] = 'cooldown'
                API_KEYS[key_to_use]['unlock_time'] = unlock_time
                print(f"\n[严重] Key ...{key_display} 触发速率限制！进入冷却状态 {cooldown_seconds} 秒...")
            else:
                print(f"\n[错误] Key ...{key_display} 发生错误: {e}。将短暂冷却5秒后重试。")
                API_KEYS[key_to_use]['status'] = 'cooldown'
                API_KEYS[key_to_use]['unlock_time'] = time.time() + 5

async def _generate_data_for_task(task_name: str, task_config: Dict, generator_config: Dict, constitution_module: Type, output_file: str):
    """为单个任务生成数据。"""
    target_samples = generator_config["TARGET_SAMPLES_PER_TASK"]
    samples_per_request = generator_config["SAMPLES_PER_REQUEST"]
    model_name = generator_config["MODEL_NAME"]
    prompt_func = getattr(constitution_module, task_config["constitution_func_name"])
    expected_keys = task_config["expected_keys"]
    
    print(f"\n--- 开始为任务生成数据: {task_name} ---")
    
    existing_samples = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_samples = sum(1 for _ in f)
    print(f"在 {output_file} 中发现 {existing_samples} 条已存在的样本。")

    if existing_samples >= target_samples:
        print(f"已达到 {target_samples} 的样本目标。跳过此任务。")
        return

    samples_to_generate = target_samples - existing_samples
    prompt = prompt_func(samples_per_request)
    
    with tqdm(total=samples_to_generate, desc=f"正在生成: {task_name}") as pbar:
        tasks_needed = (samples_to_generate + samples_per_request - 1) // samples_per_request
        worker_tasks = [_worker(prompt, output_file, pbar, model_name, expected_keys) for _ in range(tasks_needed)]
        await asyncio.gather(*worker_tasks)

def run(config_module: Type, constitution_module: Type):
    """数据生成引擎的主入口。"""
    _initialize_api_keys()
    
    raw_data_dir = config_module.DATA_DIR / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    generator_config = config_module.GENERATOR_CONFIG
    
    async def main_async():
        start_time = time.time()
        for task_name, task_config in config_module.TASKS.items():
            output_file = raw_data_dir / f"{task_name}_dataset.jsonl"
            await _generate_data_for_task(task_name, task_config, generator_config, constitution_module, output_file)
        end_time = time.time()
        print(f"\n所有数据生成任务完成，总耗时: {end_time - start_time:.2f} 秒。")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())