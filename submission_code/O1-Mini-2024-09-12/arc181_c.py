# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    P = list(map(int, N_and_rest[1:N+1]))
    Q = list(map(int, N_and_rest[N+1:2*N+1]))
    
    # Compute inverse permutations
    P_rev = [0]*(N+1)
    Q_rev = [0]*(N+1)
    
    for idx, val in enumerate(P, 1):
        P_rev[val] = idx
    for idx, val in enumerate(Q, 1):
        Q_rev[val] = idx
    
    # Build the grid
    grid = []
    for i in range(1, N+1):
        row = []
        for j in range(1, N+1):
            if P_rev[i] > Q_rev[j]:
                row.append('1')
            else:
                row.append('0')
        grid.append(''.join(row))
    
    # Print the grid
    for row in grid:
        print(row)

if __name__ == "__main__":
    main()