prompt_template = """
<|user|>
{question_text}
Options:
A. {text_a}
B. {text_b}
C. {text_c}
D. {text_d}

Return the letter of the correct answer.<|end|> 
<|assistant|>
"""

prompt_template_boolean = """
<|user|>
{question_text}
Options:
A. true
B. false

Return the letter of the correct answer.<|end|> 
<|assistant|>
"""

cot_template = """
False or True or not ( True and not True ) is 

Solving one term at a time:
( True and not True ) is False
not ( True and not True ) is True
False or True or not ( True and not True ) is True

{question_text}
"""

cot_template = """
<|user|>
False or True or not ( True and not True ) is<|end|>  

<|assistant|>
Solving one term at a time:
( True and not True ) is False
not ( True and not True ) is True
False or True or not ( True and not True ) is True
The answer is True.

<|user|>
{question_text}<|end|>  

<|assistant|>
"""