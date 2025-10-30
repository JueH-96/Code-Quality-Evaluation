def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + H):
        grid.append(data[i].strip())
    
    top = H
    bottom = -1
    left = W
    right = -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < top:
                    top = i
                if i > bottom:
                    bottom = i
                if j < left:
                    left = j
                if j > right:
                    right = j
    
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                print(f"{i + 1} {j + 1}")
                return

if __name__ == "__main__":
    main()