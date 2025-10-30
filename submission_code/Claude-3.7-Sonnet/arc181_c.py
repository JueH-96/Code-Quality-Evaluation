def solve_grid(N, P, Q):
    # Convert 1-indexed permutations to 0-indexed for ease of calculation
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Calculate the rank (or position) of each element in the permutations
    rank_P = [0] * N
    rank_Q = [0] * N
    
    for i in range(N):
        rank_P[P[i]] = i
        rank_Q[Q[i]] = i
    
    # Fill the grid
    grid = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if rank_P[i] + rank_Q[j] > N - 1:
                grid[i][j] = 1
    
    return grid

def main():
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    grid = solve_grid(N, P, Q)
    
    # Print the solution
    for row in grid:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    main()