import sys
from collections import defaultdict

# parse input data from stdin
N, _T = input().split()
S = [input().strip() for _ in range(int(N))]

# unique strings in S
unique_S = sorted(set(S))

# preprocess T and S
T_map = defaultdict(list)
T = list(_T)
T_len = len(T)

# build T_map
for i, s in enumerate(unique_S):
    dp = [0] * (T_len + 1)
    
    for ch in s:
        for ti in range(T_len - 1, -1, -1):
            if ch == T[ti]:
                dp[ti + 1] += dp[ti]
        dp[ti + 1] += 1
    
    for ti in range(T_len + 1):
        T_map[T_len - ti].append(dp[T_len - ti])

# calculate results for all pairs
T_map[-1] = [1] * len(unique_S)
result = 0
for i, si in enumerate(S):
    curr = 0
    for ch in si:
        curr = T_map[curr][unique_S.index(ch)]
    result += curr * S.count(si)

# adjust for duplicate strings in S
for si in unique_S:
    c = S.count(si)
    result -= (c - 1) * T_map[0][unique_S.index(si)] * c

print(result)