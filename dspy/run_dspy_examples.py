"""
DSPy示例运行脚本
提供统一的入口来运行各种DSPy示例
"""

import sys
import os
import argparse
from typing import Optional

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'dspy',
        'openai',
        'python-dotenv'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r dspy_requirements.txt")
        return False

    print("✅ 所有依赖包已安装")
    return True

def check_env_file():
    """检查.env文件"""
    if not os.path.exists('.env'):
        print("⚠️  未找到.env文件")
        print("请复制.env.example为.env并配置API密钥")
        return False

    # 检查是否有API密钥
    from dotenv import load_dotenv
    load_dotenv()

    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }

    available_keys = [key for key, value in api_keys.items() if value]

    if not available_keys:
        print("❌ 未配置任何API密钥")
        print("请在.env文件中配置至少一个API密钥")
        return False

    print(f"✅ 找到API密钥: {', '.join(available_keys)}")
    return True

def run_example(example_type: str, interactive: bool = False):
    """运行指定类型的示例"""
    examples = {
        'basic': 'dspy_basic_qa',
        'cot': 'dspy_cot_examples',
        'rag': 'dspy_rag_examples',
        'all': 'all'
    }

    if example_type not in examples:
        print(f"❌ 未知的示例类型: {example_type}")
        print(f"可用的示例类型: {', '.join(examples.keys())}")
        return

    if example_type == 'all':
        run_all_examples(interactive)
        return

    module_name = examples[example_type]

    try:
        if interactive:
            print(f"\n🚀 准备运行{module_name}示例...")
            user_input = input("按Enter继续，或输入'q'退出: ").strip()
            if user_input.lower() == 'q':
                return

        print(f"\n🎯 运行{module_name}示例")
        print("=" * 60)

        # 动态导入并运行模块
        if example_type == 'basic':
            from dspy_basic_qa import main as basic_main
            basic_main()
        elif example_type == 'cot':
            from dspy_cot_examples import main as cot_main
            cot_main()
        elif example_type == 'rag':
            from dspy_rag_examples import main as rag_main
            rag_main()

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有示例文件都在当前目录中")
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")

def run_all_examples(interactive: bool = False):
    """运行所有示例"""
    examples = [
        ('basic', '基础问答系统'),
        ('cot', '链式思考示例'),
        ('rag', '检索增强生成示例')
    ]

    print("🎉 将依次运行所有DSPy示例")
    print("=" * 60)

    for example_type, description in examples:
        if interactive:
            print(f"\n准备运行: {description}")
            user_input = input("按Enter继续，或输入's'跳过此示例: ").strip()
            if user_input.lower() == 's':
                continue

        print(f"\n{'='*20} {description} {'='*20}")
        try:
            run_example(example_type, interactive=False)
        except KeyboardInterrupt:
            print(f"\n用户中断了{description}的运行")
            break
        except Exception as e:
            print(f"❌ {description}运行失败: {e}")

        if interactive:
            input("按Enter继续下一个示例...")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DSPy示例运行器')
    parser.add_argument(
        'example',
        choices=['basic', 'cot', 'rag', 'all'],
        help='要运行的示例类型'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='交互式运行模式'
    )
    parser.add_argument(
        '--check-only', '-c',
        action='store_true',
        help='仅检查环境和依赖，不运行示例'
    )

    args = parser.parse_args()

    print("🤖 DSPy示例运行器")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查环境
    env_ok = check_env_file()
    if not env_ok:
        print("\n⚠️  警告: 未配置API密钥")
        print("某些示例可能无法正常运行，但可以查看代码结构")

        if not args.check_only:
            user_input = input("是否继续运行示例？[y/N]: ").strip()
            if user_input.lower() != 'y':
                sys.exit(1)

    if args.check_only:
        print("\n✅ 环境检查完成")
        sys.exit(0)

    # 运行示例
    try:
        run_example(args.example, args.interactive)
    except KeyboardInterrupt:
        print("\n\n👋 用户中断了程序运行")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()