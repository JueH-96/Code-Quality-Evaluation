from itertools import product

def eval_expr(s):
    try:
        return int(s.replace('*', ')*('))
    except:
        return None

def solve(n):
    for l in range(1, 1000 // 2 + 1):
        for p in product('123456789*', repeat=l):
            s = ''.join(p)
            s = s + s[-2::-1]
            if eval_expr(s) == n:
                return s
    return -1

n = int(input())
print(solve(n))