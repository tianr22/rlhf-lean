import re


def replace_first_lean4_code(original: str, new_code: str) -> str:
    """
    替换字符串中第一个 Lean4 代码块的内容
    """
    # 匹配第一个 Lean4 代码块的正则表达式
    pattern = r"```(?:lean4|language-lean4)(.*?```)"

    # 使用非贪婪匹配，确保匹配第一个代码块
    match = re.search(pattern, original, re.DOTALL)

    if not match:
        return original  # 未找到代码块时返回原字符串

    # 计算代码块的起止位置
    start_pos = match.start()
    end_pos = match.end()

    # 提取代码块前的文本
    prefix = original[:start_pos]

    # 提取代码块后的文本
    suffix = original[end_pos:]

    # 构建新代码块（保留原语言标识）
    new_block = f"```lean4\n{new_code.strip()}\n```"

    # 组合新字符串
    return prefix + new_block + suffix


original_str = """
思考过程：
<think>
我们需要证明这个定理...
</think>

首先尝试以下代码：
```lean4
theorem test : True := by trivial
```
"""

print(replace_first_lean4_code(original_str, "theorem test : True := by sorry"))
