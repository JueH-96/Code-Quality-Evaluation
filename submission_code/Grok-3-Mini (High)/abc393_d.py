import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
pos_list = [idx for idx, char in enumerate(S, start=1) if char == '1']
K = len(pos_list)
C_val = [pos_list[idx] - idx for idx in range(K)]
sort_C = sorted(C_val)
median_val = sort_C[K // 2]
min_swaps = sum(abs(c - median_val) for c in sort_C)
print(min_swaps)