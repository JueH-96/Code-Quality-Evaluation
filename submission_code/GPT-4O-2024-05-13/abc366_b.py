# YOUR CODE HERE
def vertical_writing():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:]
    
    M = max(len(s) for s in S)
    
    # Initialize the result matrix with '*' characters
    result = [['*' for _ in range(N)] for _ in range(M)]
    
    # Fill the result matrix with the given strings
    for i in range(N):
        for j in range(len(S[i])):
            result[j][N-i-1] = S[i][j]
    
    # Convert the result matrix to the required format
    for row in result:
        print(''.join(row).rstrip('*'))

vertical_writing()