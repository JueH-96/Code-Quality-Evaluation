# YOUR CODE HERE
def solve(N, P, Q):
    grid = [['0' for _ in range(N)] for _ in range(N)]
    
    # Helper function to get the inverse permutation
    def inverse_perm(perm):
        inv = [0] * (N + 1)
        for i, p in enumerate(perm, 1):
            inv[p] = i
        return inv
    
    P_inv = inverse_perm(P)
    Q_inv = inverse_perm(Q)
    
    for i in range(N):
        for j in range(N):
            if P_inv[i+1] < P_inv[j+1] and Q_inv[i+1] < Q_inv[j+1]:
                grid[i][j] = '1'
    
    return [''.join(row) for row in grid]

# Read input
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Solve and print output
result = solve(N, P, Q)
for row in result:
    print(row)