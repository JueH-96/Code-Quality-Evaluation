def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:]
    
    min_operations = float('inf')
    found = False
    
    # Check horizontally
    for i in range(H):
        for j in range(W - K + 1):
            count_dots = 0
            possible = True
            for k in range(K):
                if grid[i][j + k] == 'x':
                    possible = False
                    break
                elif grid[i][j + k] == '.':
                    count_dots += 1
            if possible:
                found = True
                min_operations = min(min_operations, count_dots)
    
    # Check vertically
    for j in range(W):
        for i in range(H - K + 1):
            count_dots = 0
            possible = True
            for k in range(K):
                if grid[i + k][j] == 'x':
                    possible = False
                    break
                elif grid[i + k][j] == '.':
                    count_dots += 1
            if possible:
                found = True
                min_operations = min(min_operations, count_dots)
    
    if found:
        print(min_operations)
    else:
        print(-1)

if __name__ == "__main__":
    main()