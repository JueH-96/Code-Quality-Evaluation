# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    p = int(data[1])
    
    A = []
    index = 2
    for i in range(N):
        A.append([int(data[index + j]) for j in range(N)])
        index += N
    
    # Find all zero positions
    zero_positions = []
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                zero_positions.append((i, j))
    
    K = len(zero_positions)
    
    # Initialize the result matrix
    result = [[0] * N for _ in range(N)]
    
    # Iterate over all possible replacements for zeros
    from itertools import product
    
    for replacements in product(range(1, p), repeat=K):
        B = [row[:] for row in A]
        for idx, (i, j) in enumerate(zero_positions):
            B[i][j] = replacements[idx]
        
        # Add B to the result
        for i in range(N):
            for j in range(N):
                result[i][j] = (result[i][j] + B[i][j]) % p
    
    # Print the result matrix
    for i in range(N):
        print(' '.join(map(str, result[i])))