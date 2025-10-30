from itertools import product

def evaluate(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return eval(s)

def check_palindrome(s: str) -> bool:
    return s == s[::-1]

def find_palindrome_expression(target: int) -> str:
    digits = '123456789'
    for n in range(1, 501):
        length = 2*n 
        for op in product("1*+", repeat=n-1):
            formula = [digits[0]] 
            for o, d in zip(op, digits[1:n]):
                formula.append(o+d)
            formula.append(digits[n])

            s = ''.join(formula) + ''.join([x for x in reversed(digits[:n])] + [x for x in reversed(list(op))])
            if evaluate(s) == target and check_palindrome(s):
                return s
    return "-1"

N = int(input())
print(find_palindrome_expression(N))