def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    grid_A = [input().strip() for _ in range(N)]
    grid_B = [input().strip() for _ in range(N)]
    
    # Iterate through the grids and find the only differing cell.
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                print(i + 1, j + 1)
                return

if __name__ == '__main__':
    main()