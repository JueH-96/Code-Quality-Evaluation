# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
S = data[2]

# Find all 1-blocks
blocks = []
i = 0
while i < N:
    if S[i] == '1':
        start = i
        while i < N and S[i] == '1':
            i += 1
        blocks.append((start, i - 1))
    i += 1

# Move the K-th 1-block to immediately after the (K-1)-th 1-block
k_start, k_end = blocks[K-1]
k1_start, k1_end = blocks[K-2]

# Create the new string T
T = list(S)
T[k1_end + 1:k1_end + 1 + (k_end - k_start + 1)] = '1' * (k_end - k_start + 1)
T[k_start:k_end + 1] = '0' * (k_end - k_start + 1)

print(''.join(T))