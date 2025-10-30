MOD = 998244353

s = input().strip()
n = len(s)

# Initialize DP with balance 0 having 1 way
prev_dp = {0: 1}

for c in s:
    curr_dp = {}
    for balance, count in prev_dp.items():
        # Consider '(' if current character is '(' or '?'
        if c in {'(', '?'}:
            new_balance = balance + 1
            if new_balance >= 0:
                curr_dp[new_balance] = (curr_dp.get(new_balance, 0) + count) % MOD
        # Consider ')' if current character is ')' or '?'
        if c in {')', '?'}:
            new_balance = balance - 1
            if new_balance >= 0:
                curr_dp[new_balance] = (curr_dp.get(new_balance, 0) + count) % MOD
    prev_dp = curr_dp
    # Early exit if no possible ways
    if not prev_dp:
        break

print(prev_dp.get(0, 0) % MOD)