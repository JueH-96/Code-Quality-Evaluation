import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Create position lists for each value
pos_list = [[] for _ in range(N + 1)]
for i in range(N):
    val = A[i]
    pos_list[val].append(i + 1)  # Store 1-based positions

# Compute Sum1
Sum1 = 0
total_subarrays = (N * (N + 1)) // 2
for x in range(1, N + 1):
    pos = pos_list[x]
    if pos:  # If x appears
        m = len(pos)
        sum_no_x = 0
        # Left gap
        left_gap = pos[0] - 1
        sum_no_x += (left_gap * (left_gap + 1)) // 2
        # Right gap
        right_gap = N - pos[-1]
        sum_no_x += (right_gap * (right_gap + 1)) // 2
        # Between gaps
        for i in range(m - 1):
            gap_len = pos[i + 1] - pos[i] - 1
            sum_no_x += (gap_len * (gap_len + 1)) // 2
        num_with_x = total_subarrays - sum_no_x
        Sum1 += num_with_x

# Compute Sum2
Sum2 = 0
for k in range(1, N):  # k from 1 to N-1
    if pos_list[k] and pos_list[k + 1]:  # Both k and k+1 appear
        min_k = pos_list[k][0]
        max_k = pos_list[k][-1]
        min_k1 = pos_list[k + 1][0]
        max_k1 = pos_list[k + 1][-1]
        R_min_k = max(min_k, min_k1)
        L_max_k = min(max_k, max_k1)
        # Compute number of subarrays containing both k and k+1
        num_L1 = min(L_max_k, R_min_k)
        sum_part1 = num_L1 * (N - R_min_k + 1)
        if R_min_k + 1 <= L_max_k:
            len2 = L_max_k - R_min_k
            first_val = N - R_min_k
            last_val = N - L_max_k + 1
            sum_part2 = (len2 * (first_val + last_val)) // 2
        else:
            sum_part2 = 0
        num_both_k = sum_part1 + sum_part2
        Sum2 += num_both_k

# Compute answer
answer = Sum1 - Sum2
print(answer)