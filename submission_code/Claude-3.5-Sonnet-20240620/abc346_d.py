# YOUR CODE HERE
N = int(input())
S = input()
C = list(map(int, input().split()))

def min_cost_to_good_string(N, S, C):
    # Initialize dp arrays
    dp0 = [0] * (N + 1)  # dp0[i]: min cost to make S[:i] good with S[i-1] = 0
    dp1 = [0] * (N + 1)  # dp1[i]: min cost to make S[:i] good with S[i-1] = 1
    
    for i in range(2, N + 1):
        # Case 1: Make S[i-1] = 0
        cost_to_0 = 0 if S[i-1] == '0' else C[i-1]
        dp0[i] = min(dp0[i-1] + cost_to_0, dp1[i-1] + cost_to_0)
        
        # Case 2: Make S[i-1] = 1
        cost_to_1 = C[i-1] if S[i-1] == '0' else 0
        dp1[i] = min(dp0[i-1] + cost_to_1, dp1[i-1] + cost_to_1)
        
        # Ensure at least one pair of adjacent same characters
        if i > 2:
            dp0[i] = min(dp0[i], dp1[i-2] + cost_to_0)
            dp1[i] = min(dp1[i], dp0[i-2] + cost_to_1)
    
    return min(dp0[N], dp1[N])

result = min_cost_to_good_string(N, S, C)
print(result)