def is_palindrome(s):
    return s == s[::-1]

N, K = map(int, input().split())
S = input().strip()

MOD = 998244353

from collections import defaultdict

dp = {(): 1}

for i in range(N):
    new_dp = defaultdict(int)
    
    for state, count in dp.items():
        candidates = ['A', 'B'] if S[i] == '?' else [S[i]]
        
        for char in candidates:
            new_state = state + (char,)
            
            if len(new_state) >= K:
                last_k = new_state[-K:]
                if is_palindrome(''.join(last_k)):
                    continue
            
            if len(new_state) > K - 1:
                new_state = new_state[-(K-1):]
            
            new_dp[new_state] = (new_dp[new_state] + count) % MOD
    
    dp = new_dp

print(sum(dp.values()) % MOD)