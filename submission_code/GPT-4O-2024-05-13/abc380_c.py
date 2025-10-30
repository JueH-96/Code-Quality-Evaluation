# YOUR CODE HERE
def move_kth_1_block(N, K, S):
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
    
    # Get the positions of the K-th and (K-1)-th 1-blocks
    l_k, r_k = blocks[K-1]
    l_k1, r_k1 = blocks[K-2]
    
    # Create the new string T
    T = list(S)
    
    # Move the K-th 1-block to immediately after the (K-1)-th 1-block
    T[r_k1+1:r_k1+1+(r_k-l_k+1)] = ['1'] * (r_k-l_k+1)
    T[r_k1+1+(r_k-l_k+1):r_k+1] = ['0'] * (r_k-l_k+1)
    
    print(''.join(T))

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
S = data[2]

# Call the function
move_kth_1_block(N, K, S)