# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N]
    
    # Precompute the positions of M, E, X
    M_indices = [i for i, c in enumerate(S) if c == 'M']
    E_indices = [i for i, c in enumerate(S) if c == 'E']
    X_indices = [i for i, c in enumerate(S) if c == 'X']
    
    # Precompute the counts of A[i] for each position
    # For M, E, X, we need to count the number of 0, 1, 2 in their respective positions
    # But since we need to find all (i,j,k) where S_i = M, S_j = E, S_k = X, and i < j < k
    # We can iterate over all possible E positions and count the number of M before and X after
    
    # Precompute prefix sums for M and suffix sums for X
    # prefix_M[x][i]: number of M before position i with A[i] == x
    # suffix_X[x][i]: number of X after position i with A[i] == x
    
    # Initialize prefix_M
    prefix_M = [ [0]*(N+1) for _ in range(3) ]
    for x in range(3):
        cnt = 0
        for i in range(N):
            if S[i] == 'M' and A[i] == x:
                cnt += 1
            prefix_M[x][i+1] = cnt
    
    # Initialize suffix_X
    suffix_X = [ [0]*(N+1) for _ in range(3) ]
    for x in range(3):
        cnt = 0
        for i in range(N-1, -1, -1):
            if S[i] == 'X' and A[i] == x:
                cnt += 1
            suffix_X[x][i] = cnt
    
    # Now, for each E position j, we need to find the number of M before j and X after j
    # and for each combination of A[i], A[j], A[k], compute mex and multiply by the counts
    
    total = 0
    for j in E_indices:
        a_j = A[j]
        # Count the number of M before j with A[i] == x
        # and the number of X after j with A[k] == y
        # for all x and y
        # Then, for each (x, y), compute mex(x, a_j, y) and multiply by the counts
        for x in range(3):
            cnt_M_x = prefix_M[x][j]
            for y in range(3):
                cnt_X_y = suffix_X[y][j+1]
                if cnt_M_x == 0 or cnt_X_y == 0:
                    continue
                # Compute mex(x, a_j, y)
                mex = 0
                while mex in {x, a_j, y}:
                    mex += 1
                total += mex * cnt_M_x * cnt_X_y
    
    print(total)

if __name__ == "__main__":
    main()