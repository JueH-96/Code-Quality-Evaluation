# YOUR CODE HERE
N, M = map(int, input().split())

# Read grid S
S = []
for i in range(N):
    S.append(input().strip())

# Read grid T
T = []
for i in range(M):
    T.append(input().strip())

# Try all possible positions where T could fit in S
for a in range(1, N - M + 2):  # a from 1 to N-M+1
    for b in range(1, N - M + 2):  # b from 1 to N-M+1
        # Check if T matches the subgrid of S starting at (a,b)
        match = True
        for i in range(M):
            for j in range(M):
                # Convert to 0-indexed for array access
                if S[a-1+i][b-1+j] != T[i][j]:
                    match = False
                    break
            if not match:
                break
        
        if match:
            print(a, b)
            break
    if match:
        break