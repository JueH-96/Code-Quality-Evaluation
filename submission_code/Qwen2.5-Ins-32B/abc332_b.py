# YOUR CODE HERE
def solve(K, G, M):
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:
            glass = 0
        elif mug == 0:
            mug = M
        else:
            transfer = min(G - glass, mug)
            glass += transfer
            mug -= transfer
    
    print(glass, mug)

# Reading input from stdin
K, G, M = map(int, input().split())

# Solving the problem and writing the answer to stdout
solve(K, G, M)