# YOUR CODE HERE
N, K = map(int, input().split())
S = input().strip()

# Find all 1-blocks
blocks = []
i = 0
while i < N:
    if S[i] == '1':
        start = i
        while i < N and S[i] == '1':
            i += 1
        end = i - 1
        blocks.append((start, end))
    else:
        i += 1

# Get the (K-1)-th and K-th blocks (0-indexed, so K-2 and K-1)
prev_block_end = blocks[K-2][1]  # r_{K-1}
curr_block_start = blocks[K-1][0]  # l_K
curr_block_end = blocks[K-1][1]    # r_K
curr_block_length = curr_block_end - curr_block_start + 1

# Build the result string T
T = ['0'] * N

# T_i = S_i for 1 <= i <= r_{K-1}
for i in range(prev_block_end + 1):
    T[i] = S[i]

# T_i = 1 for r_{K-1} + 1 <= i <= r_{K-1} + (r_K - l_K) + 1
for i in range(prev_block_end + 1, prev_block_end + 1 + curr_block_length):
    T[i] = '1'

# T_i = 0 for r_{K-1} + (r_K - l_K) + 2 <= i <= r_K
for i in range(prev_block_end + 1 + curr_block_length, curr_block_end + 1):
    T[i] = '0'

# T_i = S_i for r_K + 1 <= i <= N
for i in range(curr_block_end + 1, N):
    T[i] = S[i]

print(''.join(T))