def move_kth_1_block(N, K, S):
    # Step 1: Identify all 1-blocks in the string S
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
    
    # Step 2: Move the K-th 1-block to immediately after the (K-1)-th 1-block
    k_block_start, k_block_end = blocks[K - 1]
    k_minus_1_block_start, k_minus_1_block_end = blocks[K - 2]
    
    # Create a new list to build the result
    result = list(S)
    
    # Remove the K-th block from its original position
    for i in range(k_block_start, k_block_end + 1):
        result[i] = '0'
    
    # Calculate the new position for the K-th block
    new_position_start = k_minus_1_block_end + 1
    new_position_end = new_position_start + (k_block_end - k_block_start)
    
    # Insert the K-th block in the new position
    for i in range(new_position_start, new_position_end + 1):
        result[i] = '1'
    
    # Convert the list back to a string
    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
data = input().strip().splitlines()
N, K = map(int, data[0].split())
S = data[1]

# Get the result and print it
result = move_kth_1_block(N, K, S)
print(result)