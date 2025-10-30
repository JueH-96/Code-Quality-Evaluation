import sys
def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # Find bounding box of all existing black cells
    min_i, max_i = H, -1
    min_j, max_j = W, -1
    for i in range(H):
        for j, ch in enumerate(grid[i]):
            if ch == '#':
                if i < min_i: min_i = i
                if i > max_i: max_i = i
                if j < min_j: min_j = j
                if j > max_j: max_j = j
    
    # Check inside the bounding box for any white cells,
    # which would make it impossible to fill into a full rectangle.
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if grid[i][j] == '.':
                print("No")
                return
    
    print("Yes")

# Call the main function
main()