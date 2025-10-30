# YOUR CODE HERE
def find_subgrid_position():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    S = data[1:N+1]
    T = data[N+1:N+1+M]
    
    # Iterate over all possible top-left corners of the MxM subgrid in the NxN grid
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            # Check if the subgrid starting at (a, b) matches T
            match = True
            for i in range(M):
                for j in range(M):
                    if S[a + i][b + j] != T[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                # Output the 1-based indices
                print(a + 1, b + 1)
                return

# Call the function to execute the solution
find_subgrid_position()