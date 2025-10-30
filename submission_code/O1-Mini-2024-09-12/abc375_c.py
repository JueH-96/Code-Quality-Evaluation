import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    grid = [list(data[i+1]) for i in range(N)]
    
    for i in range(N//2):
        l = i
        r = N -1 -i
        for k in range(r - l):
            # Perform 4-way swap
            tmp = grid[l][l +k]
            grid[l][l +k] = grid[r -k][l]
            grid[r -k][l] = grid[r][r -k]
            grid[r][r -k] = grid[l +k][r]
            grid[l +k][r] = tmp
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()