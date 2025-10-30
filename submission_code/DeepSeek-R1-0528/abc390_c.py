def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
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
                print("No")
                return
                
    for i in range(H):
        for j in range(W):
            if i < top or i > bottom or j < left or j > right:
                if grid[i][j] == '#':
                    print("No")
                    return
                    
    print("Yes")

if __name__ == "__main__":
    main()