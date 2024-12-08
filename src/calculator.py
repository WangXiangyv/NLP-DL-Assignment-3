''' 
Adapted from 
https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
MIT License
Copyright (c) 2021 OpenAI
'''
from contextlib import contextmanager
import signal
import ast
import regex
from typing import Tuple

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

def verify_math_expression(expr: str) -> Tuple[bool, bool]:
    """
    Check whether a string is a valid math expression.
    """
    formulas = expr.split("=")
    is_valid = True
    is_correct = True
    value = None
    for f in formulas:
        val = eval_with_timeout(f)
        if val is not None:
            if value is None:
                value = val
            elif value != val:
                is_correct = False
        else:
            is_valid = False
    return is_valid, is_correct


def extract_math_expressions(text, left_token:str = "<<", right_token:str=">>"):
    """
    Extract expressions from text answer
    """
    pattern = f'{left_token}(.*?){right_token}'
    expressions = regex.findall(pattern, text)
    
    return expressions

class Calculator:
    """
    Calculate based evaluator for LLM on gasm8k
    """
    def __init__(
        self,
        valid_threshold: float = 0,
        acc_threshold: float = 0,
    ):
        assert valid_threshold >= 0 and valid_threshold <= 1
        assert acc_threshold >= 0 and acc_threshold <= 1
        self.valid_threshold = valid_threshold
        self.acc_threshold = acc_threshold
    
    def eval_answer(self, text: str) -> bool:
        expressions = extract_math_expressions(text)
        if len(expressions) == 0:
            return False
        expr_num = len(expressions)
        valid_expr_num = 0
        correct_expr_num = 0
        for expr in expressions:
            is_valid, is_correct = verify_math_expression(expr)
            if is_valid:
                valid_expr_num += 1
            if is_correct:
                correct_expr_num += 1
        valid_rate = valid_expr_num / expr_num
        correct_rate = correct_expr_num / expr_num
        # return expr_num, valid_expr_num, correct_expr_num
        if valid_rate >= self.valid_threshold and correct_rate >= self.acc_threshold:
            return True
        else:
            return False
        
if __name__ == "__main__":
    C = Calculator()
    print(C.eval_answer("In 8 weeks, Bailey receives $5 * 8 = $<<5*8=40>>40 in allowance\nBailey started with $100 - $40 = $<<100-40=60>>60"))