def create_grid(N, P, Q):
    # Initialize the grid with zeros
    grid = [['0'] * N for _ in range(N)]
    
    # Fill the grid with '1's diagonally based on P and Q
    for i in range(N):
        # Find the position in P and Q
        p_pos = P.index(i + 1)
        q_pos = Q.index(i + 1)
        
        # Fill the grid with '1's up to the (p_pos, q_pos)
        for j in range(p_pos + 1):
            grid[j][q_pos] = '1'
        for j in range(q_pos + 1):
            grid[p_pos][j] = '1'
    
    return grid

def main():
    N = int(input().strip())
    P = list(map(int, input().strip().split()))
    Q = list(map(int, input().strip().split()))

    grid = create_grid(N, P, Q)
    
    # Print the grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()