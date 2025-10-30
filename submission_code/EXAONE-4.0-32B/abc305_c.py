def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + H):
        grid.append(data[i].strip())
    
    min_i = H
    max_i = -1
    min_j = W
    max_j = -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_i:
                    min_i = i
                if i > max_i:
                    max_i = i
                if j < min_j:
                    min_j = j
                if j > max_j:
                    max_j = j
    
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if grid[i][j] == '.':
                print(f"{i + 1} {j + 1}")
                return

if __name__ == "__main__":
    main()