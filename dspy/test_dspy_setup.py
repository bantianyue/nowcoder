"""
DSPy环境测试脚本
用于测试DSPy安装和基本功能
"""

import sys
import os
import traceback

def test_imports():
    """测试包导入"""
    print("🔍 测试包导入...")

    packages = [
        ('dspy', 'DSPy核心包'),
        ('dspy.OpenAI', 'OpenAI集成'),
        ('dspy.ChainOfThought', '链式思考模块'),
        ('dspy.Prediction', '预测模块'),
        ('dspy.Example', '示例模块'),
        ('dotenv', '环境变量管理'),
        ('numpy', 'NumPy数值计算'),
        ('sklearn', 'Scikit-learn机器学习库')
    ]

    success_count = 0

    for package_name, description in packages:
        try:
            exec(f"import {package_name}")
            print(f"  ✅ {description} ({package_name})")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {description} ({package_name}): {e}")

    print(f"\n导入测试结果: {success_count}/{len(packages)} 成功")
    return success_count == len(packages)

def test_dspy_basic_functionality():
    """测试DSPy基本功能"""
    print("\n🧪 测试DSPy基本功能...")

    try:
        import dspy
        from dspy import ChainOfThought, Example, Prediction

        # 测试创建简单的ChainOfThought
        print("  📝 测试ChainOfThought创建...")
        cot = ChainOfThought("question -> answer")
        print("    ✅ ChainOfThought创建成功")

        # 测试Example和Prediction
        print("  📊 测试Example和Prediction...")
        example = Example(question="测试问题", answer="测试答案")
        prediction = Prediction(answer="预测答案")
        print("    ✅ Example和Prediction创建成功")

        # 测试模块结构
        print("  🏗️ 测试模块结构...")
        assert hasattr(dspy, 'settings'), "DSPy缺少settings模块"
        assert hasattr(dspy, 'Module'), "DSPy缺少Module基类"
        print("    ✅ 模块结构完整")

        return True

    except Exception as e:
        print(f"    ❌ 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_env_loading():
    """测试环境变量加载"""
    print("\n🔐 测试环境变量加载...")

    try:
        from dotenv import load_dotenv
        import os

        # 测试.env文件加载
        if os.path.exists('.env'):
            load_dotenv()
            print("  ✅ .env文件加载成功")
        else:
            print("  ⚠️  .env文件不存在")

        # 检查API密钥
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
        }

        available_keys = [key for key, value in api_keys.items() if value]

        if available_keys:
            print(f"  ✅ 找到API密钥: {', '.join(available_keys)}")
        else:
            print("  ⚠️  未找到API密钥")

        return True

    except Exception as e:
        print(f"  ❌ 环境变量测试失败: {e}")
        return False

def test_optional_dependencies():
    """测试可选依赖"""
    print("\n📦 测试可选依赖...")

    optional_packages = [
        ('sentence_transformers', 'SentenceTransformers', '用于文本嵌入'),
        ('transformers', 'Transformers', 'Hugging Face transformers'),
        ('torch', 'PyTorch', '深度学习框架'),
        ('faiss', 'FAISS', '向量搜索库'),
        ('pandas', 'Pandas', '数据处理库')
    ]

    success_count = 0

    for package_name, import_name, description in optional_packages:
        try:
            exec(f"import {package_name}")
            print(f"  ✅ {description} ({package_name})")
            success_count += 1
        except ImportError:
            print(f"  ⚠️  {description} ({package_name}) - 未安装")

    print(f"\n可选依赖测试结果: {success_count}/{len(optional_packages)} 已安装")
    return success_count

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")

    required_files = [
        ('dspy_basic_qa.py', '基础问答示例'),
        ('dspy_cot_examples.py', '链式思考示例'),
        ('dspy_rag_examples.py', 'RAG示例'),
        ('run_dspy_examples.py', '运行脚本'),
        ('dspy_requirements.txt', '依赖文件'),
        ('.env.example', '环境变量示例')
    ]

    success_count = 0

    for filename, description in required_files:
        if os.path.exists(filename):
            print(f"  ✅ {description} ({filename})")
            success_count += 1
        else:
            print(f"  ❌ {description} ({filename}) - 文件不存在")

    print(f"\n文件结构测试结果: {success_count}/{len(required_files)} 文件存在")
    return success_count == len(required_files)

def main():
    """主测试函数"""
    print("🧪 DSPy环境测试")
    print("=" * 50)

    test_results = []

    # 运行所有测试
    test_results.append(("包导入测试", test_imports()))
    test_results.append(("基本功能测试", test_dspy_basic_functionality()))
    test_results.append(("环境变量测试", test_env_loading()))
    test_results.append(("文件结构测试", test_file_structure()))

    optional_test_result = test_optional_dependencies()

    # 显示测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)

    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1

    print(f"\n总体结果: {passed_tests}/{len(test_results)} 测试通过")

    if passed_tests == len(test_results):
        print("\n🎉 所有核心测试通过！DSPy环境配置正确。")
        print("\n下一步:")
        print("1. 配置API密钥到.env文件")
        print("2. 运行: python run_dspy_examples.py --interactive")
        print("3. 选择要运行的示例类型")
    else:
        print("\n⚠️  存在测试失败，请检查上述错误信息。")
        print("\n建议:")
        print("1. 运行: pip install -r dspy_requirements.txt")
        print("2. 确保.env文件配置正确")
        print("3. 检查所有示例文件是否存在")

    # 显示可选依赖状态
    print(f"\n📦 可选依赖: {optional_test_result}/5 已安装")
    if optional_test_result < 5:
        print("注意: 某些高级功能可能需要安装额外的可选依赖")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现意外错误: {e}")
        traceback.print_exc()
        sys.exit(1)