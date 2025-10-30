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

# Find the position where T matches in S
for a in range(1, N - M + 2):  # 1-indexed, so a goes from 1 to N-M+1
    for b in range(1, N - M + 2):  # 1-indexed, so b goes from 1 to N-M+1
        # Check if T matches at position (a,b)
        match = True
        for i in range(M):
            for j in range(M):
                # Convert to 0-indexed for array access
                if S[a + i - 1][b + j - 1] != T[i][j]:
                    match = False
                    break
            if not match:
                break
        
        if match:
            print(a, b)
            exit()