import sys
import itertools

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_count = 0
    for a, b in itertools.combinations(floor_cells, 2):
        count = 0
        for c in floor_cells:
            d_a = abs(a[0] - c[0]) + abs(a[1] - c[1])
            d_b = abs(b[0] - c[0]) + abs(b[1] - c[1])
            if d_a <= D or d_b <= D:
                count += 1
        if count > max_count:
            max_count = count
    print(max_count)

if __name__ == "__main__":
    main()