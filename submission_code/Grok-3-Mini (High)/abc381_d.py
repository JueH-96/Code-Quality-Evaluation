import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Initialize answer
ans = 0

# Helper function to find the maximum number of consecutive distinct values in an arithmetic sequence
def max_distinct_count_in_seq(A_list, start_idx, step, num_elem):
    max_len = 0
    left = 0
    char_set = set()
    for right in range(num_elem):
        idx_right = start_idx + right * step
        val_right = A_list[idx_right]
        while val_right in char_set:
            idx_left = start_idx + left * step
            val_left = A_list[idx_left]
            char_set.remove(val_left)
            left += 1
        char_set.add(val_right)
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    return max_len

# Handle even parity (pair starts at even indices)
max_pos_even = (N - 2) // 2
if max_pos_even >= 0:
    pos = 0
    while pos <= max_pos_even:
        if A[2 * pos] == A[2 * pos + 1]:
            block_start = pos
            block_end = pos
            while block_end <= max_pos_even and A[2 * block_end] == A[2 * block_end + 1]:
                block_end += 1
            block_end -= 1  # Last index with true condition
            num_elem = block_end - block_start + 1
            start_idx = 2 * block_start
            max_pairs = max_distinct_count_in_seq(A, start_idx, 2, num_elem)
            ans = max(ans, 2 * max_pairs)
            pos = block_end + 1  # Skip to the end of the block
        else:
            pos += 1

# Handle odd parity (pair starts at odd indices)
max_pos_odd = (N - 3) // 2
if max_pos_odd >= 0:
    pos = 0
    while pos <= max_pos_odd:
        if A[2 * pos + 1] == A[2 * pos + 2]:
            block_start_pos = pos
            block_end_pos = pos
            while block_end_pos <= max_pos_odd and A[2 * block_end_pos + 1] == A[2 * block_end_pos + 2]:
                block_end_pos += 1
            block_end_pos -= 1  # Last position with true condition
            num_elem = block_end_pos - block_start_pos + 1
            start_idx = 2 * block_start_pos + 1
            max_pairs = max_distinct_count_in_seq(A, start_idx, 2, num_elem)
            ans = max(ans, 2 * max_pairs)
            pos = block_end_pos + 1  # Skip to the end of the block
        else:
            pos += 1

# Output the answer
print(ans)