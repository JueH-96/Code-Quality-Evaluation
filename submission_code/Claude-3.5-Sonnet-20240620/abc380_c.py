# YOUR CODE HERE
def move_block(S, K):
    N = len(S)
    blocks = []
    start = -1
    
    # Find all 1-blocks
    for i in range(N):
        if S[i] == '1' and (i == 0 or S[i-1] == '0'):
            start = i
        if S[i] == '1' and (i == N-1 or S[i+1] == '0'):
            blocks.append((start, i))
    
    # If K is greater than the number of blocks, return S unchanged
    if K > len(blocks):
        return S
    
    # Extract the K-th block
    k_start, k_end = blocks[K-1]
    k_block = S[k_start:k_end+1]
    
    # Remove the K-th block from S
    S_list = list(S)
    for i in range(k_start, k_end+1):
        S_list[i] = '0'
    
    # Insert the K-th block after the (K-1)-th block
    if K > 1:
        insert_pos = blocks[K-2][1] + 1
    else:
        insert_pos = 0
    
    result = ''.join(S_list[:insert_pos] + list(k_block) + S_list[insert_pos:])
    return result

# Read input
N, K = map(int, input().split())
S = input().strip()

# Solve and print output
print(move_block(S, K))