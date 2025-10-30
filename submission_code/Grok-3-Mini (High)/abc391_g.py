import sys
from collections import defaultdict

# Read input
data = sys.stdin.read().strip()
lines = data.split()
index = 0
N = int(lines[index])
index += 1
M = int(lines[index])
index += 1
S = lines[index]

MOD = 998244353

def compute_new_dp(old_dp, c, S_str):
    N_len = len(S_str)
    new_dp_list = [0] * N_len
    if S_str[0] == c:
        new_dp_list[0] = 1
    else:
        new_dp_list[0] = old_dp[0]
    
    for k in range(1, N_len):
        if S_str[k] == c:
            new_dp_list[k] = old_dp[k-1] + 1
        else:
            new_dp_list[k] = max(new_dp_list[k-1], old_dp[k])
    
    return tuple(new_dp_list)

# Initial state
initial_dp = (0,) * N
cur = {initial_dp: 1}

for _ in range(M):
    new_dict = defaultdict(int)
    for dp_tuple, f_val in cur.items():
        for c in 'abcdefghijklmnopqrstuvwxyz':
            new_dp = compute_new_dp(dp_tuple, c, S)
            new_dict[new_dp] = (new_dict[new_dp] + f_val) % MOD
    cur = dict(new_dict)

# Now cur has the dp at length M
# Sum for each LCS length
ans = [0] * (N + 1)
for dp_tuple, f_val in cur.items():
    lcs_len = dp_tuple[N-1]  # dp[N][M]
    ans[lcs_len] = (ans[lcs_len] + f_val) % MOD

# Output
print(' '.join(map(str, ans)))