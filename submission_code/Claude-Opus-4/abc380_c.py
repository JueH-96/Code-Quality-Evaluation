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
        blocks.append((start, i - 1))
    else:
        i += 1

# Convert to list for easier manipulation
T = list(S)

# Get the K-th and (K-1)-th blocks (0-indexed)
k_block = blocks[K - 1]
prev_block = blocks[K - 2]

# Length of the K-th block
k_len = k_block[1] - k_block[0] + 1

# Move the K-th block to after (K-1)-th block
# First, set the original K-th block position to 0s
for i in range(k_block[0], k_block[1] + 1):
    T[i] = '0'

# Then, set the new position to 1s
for i in range(prev_block[1] + 1, prev_block[1] + 1 + k_len):
    T[i] = '1'

print(''.join(T))