mod = 998244353

n, k = map(int, input().split())
s = input().strip()

from collections import defaultdict

def is_palindrome(sub):
    return sub == sub[::-1]

prev_dp = defaultdict(int)

# Initialize the DP for the first character
if s[0] == 'A':
    prev_dp['A'] = 1
elif s[0] == 'B':
    prev_dp['B'] = 1
else:  # s[0] is '?'
    prev_dp['A'] = 1
    prev_dp['B'] = 1

for i in range(1, n):
    current_dp = defaultdict(int)
    current_char = s[i]
    for state in prev_dp:
        count = prev_dp[state]
        if current_char == 'A' or current_char == 'B':
            next_char = current_char
            if len(state) < k - 1:
                new_state = state + next_char
                current_dp[new_state] = (current_dp[new_state] + count) % mod
            else:
                new_sub = state + next_char
                if is_palindrome(new_sub):
                    continue
                new_state = state[1:] + next_char if len(state) == k - 1 else state + next_char
                current_dp[new_state] = (current_dp[new_state] + count) % mod
        elif current_char == '?':
            for c in ['A', 'B']:
                if len(state) < k - 1:
                    new_state = state + c
                    current_dp[new_state] = (current_dp[new_state] + count) % mod
                else:
                    new_sub = state + c
                    if is_palindrome(new_sub):
                        continue
                    new_state = state[1:] + c if len(state) == k - 1 else state + c
                    current_dp[new_state] = (current_dp[new_state] + count) % mod
    prev_dp = current_dp
    if not prev_dp:
        break  # No valid ways to proceed further

total = sum(prev_dp.values()) % mod
print(total)