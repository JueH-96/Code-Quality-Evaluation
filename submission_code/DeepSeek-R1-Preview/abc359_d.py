MOD = 998244353
from collections import defaultdict

n, k = map(int, input().split())
s = input().strip()

current_dp = defaultdict(int)
current_dp[""] = 1

for i in range(n):
    next_dp = defaultdict(int)
    char = s[i]
    choices = ['A', 'B'] if char == '?' else [char]
    
    for state in current_dp:
        cnt = current_dp[state]
        for c in choices:
            new_state = (state + c)[-(k-1):] if k > 1 else ''
            # Check if the substring is a palindrome
            if len(state) == k - 1 and k >= 2:
                substr = state + c
                is_palindrome = True
                for j in range(k):
                    if substr[j] != substr[~j]:
                        is_palindrome = False
                        break
                if is_palindrome:
                    continue
            # Update next state
            next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
    current_dp = next_dp

print(sum(current_dp.values()) % MOD)