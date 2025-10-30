def check_bingo(grid, N):
    # Check rows and columns
    for i in range(N):
        if all(grid[i][j] for j in range(N)) or all(grid[j][i] for j in range(N)):
            return True
    # Check diagonals
    if all(grid[i][i] for i in range(N)) or all(grid[i][N-1-i] for i in range(N)):
        return True
    return False

def main():
    N, T = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Initialize the grid
    grid = [[False] * N for _ in range(N)]
    
    # Process each turn
    for turn, number in enumerate(A, 1):
        row = (number - 1) // N
        col = (number - 1) % N
        grid[row][col] = True
        
        # Check for bingo
        if check_bingo(grid, N):
            print(turn)
            return
    
    # If no bingo is achieved
    print(-1)

if __name__ == "__main__":
    main()