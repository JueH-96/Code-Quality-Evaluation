def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = [data[3 + i] for i in range(H)]
    
    min_ops = float('inf')
    
    # Check horizontal sequences
    if W >= K:
        for i in range(H):
            row = grid[i]
            for j in range(W - K + 1):
                window = row[j:j+K]
                dots = window.count('.')
                if dots < min_ops:
                    min_ops = dots
    
    # Check vertical sequences
    if H >= K:
        for j in range(W):
            for i in range(H - K + 1):
                dots = 0
                for k in range(K):
                    if grid[i + k][j] == '.':
                        dots += 1
                if dots < min_ops:
                    min_ops = dots
    
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == "__main__":
    main()