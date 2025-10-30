def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    # Find the bounds of the rectangle with cookies
    top = bottom = left = right = None
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if top is None:
                    top = i
                bottom = i
                if left is None or j < left:
                    left = j
                if right is None or j > right:
                    right = j
    
    # Now we need to find the one missing cookie in the rectangle
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                print(i + 1, j + 1)
                return

main()