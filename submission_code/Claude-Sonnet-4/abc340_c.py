n = int(input())

memo = {}

def f(x):
    if x == 1:
        return 0
    if x in memo:
        return memo[x]
    
    result = x + f(x // 2) + f((x + 1) // 2)
    memo[x] = result
    return result

print(f(n))