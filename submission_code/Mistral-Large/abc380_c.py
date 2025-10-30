import sys

def move_kth_1block(N, K, S):
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
        i += 1

    # Extract the K-th and (K-1)-th 1-blocks
    k_block = blocks[K-1]
    k_minus_1_block = blocks[K-2]

    # Create the resulting string T
    T = list(S)
    # Move the K-th 1-block to immediately after the (K-1)-th 1-block
    T[k_minus_1_block[1]+1:k_minus_1_block[1]+1+(k_block[1]-k_block[0]+1)] = ['1'] * (k_block[1]-k_block[0]+1)
    T[k_block[0]:k_block[1]+1] = ['0'] * (k_block[1]-k_block[0]+1)

    # Print the resulting string
    print(''.join(T))

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
S = data[2]

# Solve the problem
move_kth_1block(N, K, S)