import sys
import bisect

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))

# Compute prefix sums modulo M
prefix_mod = [0] * (N + 1)
for i in range(1, N + 1):
    prefix_mod[i] = (prefix_mod[i - 1] + A[i - 1]) % M

# Create idx_lists grouped by prefix sum modulo M
idx_lists = [[] for _ in range(M)]
for u in range(N):
    val = prefix_mod[u]
    idx_lists[val].append(u)  # Indices are appended in sorted order

# Compute first part: number of pairs (u, v) with u < v and same prefix sum modulo M
first_part = 0
for lst in idx_lists:
    f = len(lst)
    first_part += (f * (f - 1)) // 2

# Compute C value
C_val = prefix_mod[N]

# Compute second part: number of pairs (u, v) with u > v and V_u == (C + V_v) mod M
second_part = 0
for v in range(N):
    b_val = prefix_mod[v]
    a_val = (C_val + b_val) % M
    idx_list_a = idx_lists[a_val]
    num_less_equal = bisect.bisect_right(idx_list_a, v)
    num_greater = len(idx_list_a) - num_less_equal
    second_part += num_greater

# Total number of pairs (s, t) with minimum clockwise steps multiple of M
answer = first_part + second_part
print(answer)