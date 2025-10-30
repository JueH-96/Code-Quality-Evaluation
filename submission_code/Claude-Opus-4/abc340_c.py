# YOUR CODE HERE
def solve(n, memo={}):
    if n == 1:
        return 0
    if n in memo:
        return memo[n]
    
    # Cost is n plus the cost of its two halves
    result = n + solve(n // 2, memo) + solve((n + 1) // 2, memo)
    memo[n] = result
    return result

N = int(input())
print(solve(N))