# YOUR CODE HERE
N, K = map(int, input().split())
S = input().strip()

MOD = 998244353

def is_palindrome(t):
    return t == t[::-1]

# DP: dp[state] = number of ways to reach this state
# state = tuple of last min(position, K-1) characters
dp = {(): 1}

for i in range(N):
    new_dp = {}
    
    for state, count in dp.items():
        for c in ['A', 'B']:
            if S[i] != '?' and S[i] != c:
                continue
            
            # Create new state by appending c
            new_state = state + (c,)
            
            # Check if we have a K-length substring
            if len(new_state) >= K:
                # Check if the last K characters form a palindrome
                if is_palindrome(new_state[-K:]):
                    continue  # Skip this choice
                
                # Trim to keep only last K-1 characters
                new_state = new_state[-(K-1):]
            
            if new_state not in new_dp:
                new_dp[new_state] = 0
            new_dp[new_state] = (new_dp[new_state] + count) % MOD
    
    dp = new_dp

# Sum all counts
result = sum(dp.values()) % MOD
print(result)