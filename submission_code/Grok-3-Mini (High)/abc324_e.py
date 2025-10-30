import sys
import collections

def len_prefix(S_str, T_str):
    t_idx = 0
    for char in S_str:
        if t_idx < len(T_str) and char == T_str[t_idx]:
            t_idx += 1
    return t_idx

def is_subseq(S_str, T_str, start_p):
    t_idx = start_p
    for char in S_str:
        if t_idx < len(T_str) and char == T_str[t_idx]:
            t_idx += 1
    return t_idx == len(T_str)

def find_min_p(S_str, T_str):
    left = 0
    right = len(T_str)
    res = len(T_str)
    while left <= right:
        mid = (left + right) // 2
        if is_subseq(S_str, T_str, mid):
            res = mid
            right = mid - 1
        else:
            left = mid + 1
    return res

# Read input
data = sys.stdin.read().split()
index = 0
N_str = data[index]
N = int(N_str)
index += 1
T = data[index]
index += 1

# Read N strings
S_list = []
for _ in range(N):
    S = data[index]
    index += 1
    S_list.append(S)

# Get length of T
M = len(T)

# Compute len_prefix and L for each string
len_prefix_list = []
L_list = []
for S in S_list:
    lp = len_prefix(S, T)
    l_val = find_min_p(S, T)
    len_prefix_list.append(lp)
    L_list.append(l_val)

# Compute counters
prefix_counter = collections.Counter(len_prefix_list)
L_counter = collections.Counter(L_list)

# Create frequency lists
freq_L = [L_counter[l] for l in range(M + 1)]
freq_prefix = [prefix_counter[k] for k in range(M + 1)]

# Compute cumulative sum for L
cum_sum = 0
cum_L = []
for f in freq_L:
    cum_sum += f
    cum_L.append(cum_sum)

# Compute total number of pairs
total = 0
for k in range(M + 1):
    total += freq_prefix[k] * cum_L[k]

# Output the result
print(total)