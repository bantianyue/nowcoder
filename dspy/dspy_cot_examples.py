"""
DSPy 链式思考(Chain of Thought) 高级示例
展示复杂的推理和问题解决能力
"""

import dspy
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class MathProblemSolver(dspy.Module):
    """数学问题求解器，使用链式思考"""

    def __init__(self):
        super().__init__()
        # 步骤1: 理解问题
        self.understand_problem = dspy.ChainOfThought(
            "math_problem -> problem_type, given_info, goal, constraints"
        )
        # 步骤2: 制定计划
        self.plan_solution = dspy.ChainOfThought(
            "problem_type, given_info, goal, constraints -> solution_plan, steps_needed"
        )
        # 步骤3: 执行计算
        self.execute_calculation = dspy.ChainOfThought(
            "current_step, previous_results -> step_result, next_action"
        )
        # 步骤4: 验证答案
        self.verify_answer = dspy.ChainOfThought(
            "original_problem, solution_steps, final_answer -> verification, confidence_score"
        )

    def forward(self, math_problem):
        """完整的数学问题求解流程"""
        # 1. 理解问题
        understanding = self.understand_problem(math_problem=math_problem)

        # 2. 制定解决方案计划
        plan = self.plan_solution(
            problem_type=understanding.problem_type,
            given_info=understanding.given_info,
            goal=understanding.goal,
            constraints=understanding.constraints
        )

        # 3. 执行解题步骤
        solution_steps = []
        current_result = None

        steps = plan.steps_needed if hasattr(plan, 'steps_needed') else 3

        for step in range(steps):
            step_input = f"步骤 {step + 1}: {plan.solution_plan}"
            if current_result:
                step_input += f", 上一步结果: {current_result}"

            calculation = self.execute_calculation(
                current_step=step_input,
                previous_results=current_result or "无"
            )

            solution_steps.append({
                "step": step + 1,
                "description": step_input,
                "result": calculation.step_result,
                "next_action": calculation.next_action
            })

            current_result = calculation.step_result

        # 4. 验证最终答案
        verification = self.verify_answer(
            original_problem=math_problem,
            solution_steps=str(solution_steps),
            final_answer=current_result
        )

        return dspy.Prediction(
            problem_understanding=understanding,
            solution_plan=plan,
            solution_steps=solution_steps,
            final_answer=current_result,
            verification=verification
        )

class CodeGenerator(dspy.Module):
    """代码生成器，使用链式思考生成和解释代码"""

    def __init__(self):
        super().__init__()
        # 分析需求
        self.analyze_requirements = dspy.ChainOfThought(
            "code_requirements -> function_purpose, inputs, outputs, edge_cases"
        )
        # 设计算法
        self.design_algorithm = dspy.ChainOfThought(
            "function_purpose, inputs, outputs -> algorithm_steps, data_structures, complexity"
        )
        # 生成代码
        self.generate_code = dspy.ChainOfThought(
            "algorithm_steps, data_structures, programming_language -> code, explanation"
        )
        # 测试用例生成
        self.generate_tests = dspy.ChainOfThought(
            "function_purpose, inputs, outputs -> test_cases, expected_outputs"
        )

    def forward(self, code_requirements, programming_language="Python"):
        """完整的代码生成流程"""
        # 1. 分析需求
        analysis = self.analyze_requirements(code_requirements=code_requirements)

        # 2. 设计算法
        design = self.design_algorithm(
            function_purpose=analysis.function_purpose,
            inputs=analysis.inputs,
            outputs=analysis.outputs
        )

        # 3. 生成代码
        code_generation = self.generate_code(
            algorithm_steps=design.algorithm_steps,
            data_structures=design.data_structures,
            programming_language=programming_language
        )

        # 4. 生成测试用例
        tests = self.generate_tests(
            function_purpose=analysis.function_purpose,
            inputs=analysis.inputs,
            outputs=analysis.outputs
        )

        return dspy.Prediction(
            requirements_analysis=analysis,
            algorithm_design=design,
            generated_code=code_generation.code,
            code_explanation=code_generation.explanation,
            test_cases=tests.test_cases,
            expected_outputs=tests.expected_outputs
        )

class LogicalReasoning(dspy.Module):
    """逻辑推理模块，处理复杂的逻辑问题"""

    def __init__(self):
        super().__init__()
        # 识别逻辑关系
        self.identify_relations = dspy.ChainOfThought(
            "logical_problem -> entities, relationships, constraints, goal"
        )
        # 推理过程
        self.reasoning_process = dspy.ChainOfThought(
            "entities, relationships, constraints, current_state -> next_step, new_insights"
        )
        # 得出结论
        self.draw_conclusion = dspy.ChainOfThought(
            "logical_problem, reasoning_steps, all_insights -> conclusion, confidence"
        )

    def forward(self, logical_problem):
        """逻辑推理流程"""
        # 1. 识别逻辑关系
        relations = self.identify_relations(logical_problem=logical_problem)

        # 2. 多步推理
        reasoning_steps = []
        current_state = "初始状态"
        all_insights = []

        for step in range(3):  # 最多3步推理
            reasoning = self.reasoning_process(
                entities=relations.entities,
                relationships=relations.relationships,
                constraints=relations.constraints,
                current_state=current_state
            )

            reasoning_steps.append({
                "step": step + 1,
                "current_state": current_state,
                "next_step": reasoning.next_step,
                "insights": reasoning.new_insights
            })

            all_insights.append(reasoning.new_insights)
            current_state = reasoning.next_step

        # 3. 得出结论
        conclusion = self.draw_conclusion(
            logical_problem=logical_problem,
            reasoning_steps=str(reasoning_steps),
            all_insights=str(all_insights)
        )

        return dspy.Prediction(
            logical_relations=relations,
            reasoning_steps=reasoning_steps,
            conclusion=conclusion.conclusion,
            confidence=conclusion.confidence
        )

def setup_dspy():
    """配置DSPy环境"""
    lm = dspy.OpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1  # 较低的温度以获得更一致的推理
    )
    dspy.settings.configure(lm=lm)
    print("✅ DSPy环境配置完成")

def math_solver_demo():
    """数学求解器演示"""
    print("\n🔢 数学问题求解器演示")
    print("=" * 50)

    solver = MathProblemSolver()

    math_problems = [
        "一个矩形的长是宽的2倍，周长是36厘米，求这个矩形的面积。",
        "小明有50元钱，买了3支笔，每支8元，又买了2本书，每本12元，他还剩多少钱？",
        "一个等差数列的首项是3，公差是4，求第10项的值。"
    ]

    for i, problem in enumerate(math_problems, 1):
        print(f"\n数学问题 {i}: {problem}")
        try:
            result = solver(math_problem=problem)
            print(f"问题类型: {result.problem_understanding.problem_type}")
            print(f"解决方案计划: {result.solution_plan.solution_plan}")
            print(f"解题步骤:")
            for step in result.solution_steps:
                print(f"  {step['step']}. {step['description']}")
                print(f"     结果: {step['result']}")
            print(f"最终答案: {result.final_answer}")
            print(f"验证置信度: {result.verification.confidence_score}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 40)

def code_generator_demo():
    """代码生成器演示"""
    print("\n💻 代码生成器演示")
    print("=" * 50)

    generator = CodeGenerator()

    code_requirements = [
        "写一个函数，判断一个数是否为素数",
        "实现冒泡排序算法，对整数列表进行升序排列",
        "创建一个函数，计算斐波那契数列的第n项"
    ]

    for i, requirement in enumerate(code_requirements, 1):
        print(f"\n代码需求 {i}: {requirement}")
        try:
            result = generator(code_requirements=requirement)
            print(f"函数目的: {result.requirements_analysis.function_purpose}")
            print(f"算法步骤: {result.algorithm_design.algorithm_steps}")
            print(f"生成的代码:\n{result.generated_code}")
            print(f"代码解释: {result.code_explanation}")
            print(f"测试用例: {result.test_cases}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 40)

def logical_reasoning_demo():
    """逻辑推理演示"""
    print("\n🧠 逻辑推理演示")
    print("=" * 50)

    reasoner = LogicalReasoning()

    logical_problems = [
        "有三个人A、B、C，其中一个是诚实的人（总是说真话），一个是骗子（总是说假话），一个是间谍（可能说真话也可能说假话）。A说：'C是骗子'，B说：'A是诚实的人'，C说：'我是间谍'。请问谁是诚实的人？",
        "在一场比赛中，有5个队伍参赛。A队战胜了B队，B队战胜了C队，C队战胜了D队，D队战胜了E队，E队战胜了A队。如果战胜关系是可传递的，那么谁是最强的队伍？",
        "有红、蓝、绿三个盒子，其中一个有奖品。盒子上贴有标签：红盒子写着'奖品不在这里'，蓝盒子写着'奖品在红盒子里'，绿盒子写着'奖品不在这里'。已知只有一个标签是真的，奖品在哪个盒子里？"
    ]

    for i, problem in enumerate(logical_problems, 1):
        print(f"\n逻辑问题 {i}: {problem}")
        try:
            result = reasoner(logical_problem=problem)
            print(f"识别的实体: {result.logical_relations.entities}")
            print(f"推理过程:")
            for step in result.reasoning_steps:
                print(f"  步骤{step['step']}: {step['next_step']}")
                print(f"    洞察: {step['insights']}")
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 40)

def main():
    """主函数"""
    print("🧩 DSPy 链式思考高级示例")
    print("=" * 50)

    # 配置DSPy环境
    setup_dspy()

    # 运行各种演示
    math_solver_demo()
    code_generator_demo()
    logical_reasoning_demo()

    print("\n🎉 链式思考示例演示完成！")
    print("\n💡 提示:")
    print("1. 链式思考让模型能够展示推理过程")
    print("2. 多步骤分解有助于解决复杂问题")
    print("3. 验证步骤提高了答案的可靠性")

if __name__ == "__main__":
    main()