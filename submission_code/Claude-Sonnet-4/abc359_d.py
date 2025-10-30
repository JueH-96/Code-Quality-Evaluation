def solve():
    MOD = 998244353
    
    N, K = map(int, input().split())
    S = input().strip()
    
    def is_palindrome(s):
        return s == s[::-1]
    
    # DP: dp[state] = number of ways where state is the last min(i, K-1) characters
    from collections import defaultdict
    
    dp = defaultdict(int)
    dp[tuple()] = 1  # Empty state
    
    for i in range(N):
        new_dp = defaultdict(int)
        
        for state, count in dp.items():
            # Determine possible characters for position i
            if S[i] == '?':
                candidates = ['A', 'B']
            else:
                candidates = [S[i]]
            
            for char in candidates:
                # Create new state by appending char
                new_state = list(state) + [char]
                
                # If we have K characters, check palindrome
                if len(new_state) >= K:
                    last_k = new_state[-K:]
                    if is_palindrome(last_k):
                        continue  # Skip this invalid path
                
                # Keep only last K-1 characters for state
                if len(new_state) >= K:
                    final_state = tuple(new_state[-(K-1):])
                else:
                    final_state = tuple(new_state)
                
                new_dp[final_state] = (new_dp[final_state] + count) % MOD
        
        dp = new_dp
    
    # Sum all final states
    result = sum(dp.values()) % MOD
    return result

print(solve())