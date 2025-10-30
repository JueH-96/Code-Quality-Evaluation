N, K = map(int, input().split())
S = input().strip()

# Find all 1-blocks (using 1-based indexing)
blocks = []
i = 0
while i < N:
    if S[i] == '1':
        start_0based = i
        while i < N and S[i] == '1':
            i += 1
        end_0based = i - 1
        # Convert to 1-based inclusive
        start_1based = start_0based + 1
        end_1based = end_0based + 1
        blocks.append((start_1based, end_1based))
    else:
        i += 1

# Get the relevant blocks
r_prev = blocks[K-2][1]  # r_{K-1}
l_curr = blocks[K-1][0]  # l_K
r_curr = blocks[K-1][1]  # r_K

# Build result
result = [''] * N

# T_i = S_i for 1 ≤ i ≤ r_{K-1}
for i in range(1, r_prev + 1):
    result[i-1] = S[i-1]

# T_i = 1 for r_{K-1} + 1 ≤ i ≤ r_{K-1} + (r_K - l_K) + 1
for i in range(r_prev + 1, r_prev + (r_curr - l_curr) + 2):
    result[i-1] = '1'

# T_i = 0 for r_{K-1} + (r_K - l_K) + 2 ≤ i ≤ r_K
for i in range(r_prev + (r_curr - l_curr) + 2, r_curr + 1):
    result[i-1] = '0'

# T_i = S_i for r_K + 1 ≤ i ≤ N
for i in range(r_curr + 1, N + 1):
    result[i-1] = S[i-1]

print(''.join(result))