def compute(x, memo):
    if x in memo:
        return memo[x]
    a = x // 2
    b = x - a
    res = x + compute(a, memo) + compute(b, memo)
    memo[x] = res
    return res

n = int(input())
memo = {1: 0}
print(compute(n, memo))