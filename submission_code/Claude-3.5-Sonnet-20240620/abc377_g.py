# YOUR CODE HERE
def min_cost(target, previous_strings):
    dp = [float('inf')] * (len(target) + 1)
    dp[0] = 0
    
    for i in range(len(target) + 1):
        if i > 0:
            dp[i] = min(dp[i], dp[i-1] + 1)
        
        for s in previous_strings:
            if s.startswith(target[:i]):
                remaining = len(s) - i
                dp[i] = min(dp[i], remaining)
    
    return dp[-1]

N = int(input())
strings = [input().strip() for _ in range(N)]

for k in range(1, N + 1):
    target = strings[k-1]
    previous_strings = strings[:k-1]
    cost = min_cost(target, previous_strings)
    print(cost)