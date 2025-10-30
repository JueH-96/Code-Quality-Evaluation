import sys
import functools

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
K = int(data[index + 1])
index += 2
S = data[index]
MOD = 998244353
P_index = [0, 1, 2, 3, 3, 2, 1, 0]  # P_index for states 0 to 7

# DP function with memoization
@functools.lru_cache(None)
def dp(i, s, f0, f1, f2, f3):
    if i == N:
        G = 0
        for f in (f0, f1, f2, f3):
            G += (f * (f - 1)) // 2
        if G >= K:
            return 1
        else:
            return 0
    c = S[i]
    ans = 0
    if c == 'A':
        masks = [1]
    elif c == 'B':
        masks = [2]
    elif c == 'C':
        masks = [4]
    elif c == '?':
        masks = [1, 2, 4]
    for mask in masks:
        t = s ^ mask
        P_t = P_index[t]
        if P_t == 0:
            new_f0 = f0 + 1
            new_f1 = f1
            new_f2 = f2
            new_f3 = f3
        elif P_t == 1:
            new_f0 = f0
            new_f1 = f1 + 1
            new_f2 = f2
            new_f3 = f3
        elif P_t == 2:
            new_f0 = f0
            new_f1 = f1
            new_f2 = f2 + 1
            new_f3 = f3
        else:  # P_t == 3
            new_f0 = f0
            new_f1 = f1
            new_f2 = f2
            new_f3 = f3 + 1
        val = dp(i + 1, t, new_f0, new_f1, new_f2, new_f3)
        ans += val
        ans %= MOD
    return ans % MOD

# Compute the answer
answer = dp(0, 0, 1, 0, 0, 0)
print(answer)