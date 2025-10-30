def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    grid = [list(data[i].strip()) for i in range(1, N+1)]
    
    # Create the final grid based on the mapping B[x][y] = A[N-1-y][x]
    final_grid = [[''] * N for _ in range(N)]
    for x in range(N):
        for y in range(N):
            final_grid[x][y] = grid[N-1-y][x]
    
    # Print the final grid
    for row in final_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()