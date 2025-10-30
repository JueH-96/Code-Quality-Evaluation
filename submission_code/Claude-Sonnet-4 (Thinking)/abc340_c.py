def solve(n):
    memo = {}
    
    def helper(x):
        if x in memo:
            return memo[x]
        if x == 1:
            result = 0
        else:
            result = x + helper(x // 2) + helper((x + 1) // 2)
        memo[x] = result
        return result
    
    return helper(n)

n = int(input())
print(solve(n))