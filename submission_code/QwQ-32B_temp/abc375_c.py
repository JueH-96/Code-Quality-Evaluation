import sys

def main():
    N = int(sys.stdin.readline().strip())
    original = [sys.stdin.readline().strip() for _ in range(N)]
    new_grid = [['.' for _ in range(N)] for _ in range(N)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            i_max = min(x, y, N+1 -x, N+1 -y)
            mod = i_max % 4
            if mod == 0:
                new_x, new_y = x, y
            elif mod == 1:
                new_x, new_y = y, N+1 -x
            elif mod == 2:
                new_x, new_y = N+1 -x, N+1 -y
            else:  # mod == 3
                new_x, new_y = N+1 -y, x
            new_grid[new_x-1][new_y-1] = original[x-1][y-1]
    
    for row in new_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()