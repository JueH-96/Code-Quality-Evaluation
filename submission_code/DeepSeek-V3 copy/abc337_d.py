def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:3+H]
    
    min_ops = float('inf')
    
    # Check horizontal sequences
    for i in range(H):
        row = grid[i]
        for j in range(W - K + 1):
            segment = row[j:j+K]
            if 'x' in segment:
                continue
            count = segment.count('.')
            if count < min_ops:
                min_ops = count
    
    # Check vertical sequences
    for j in range(W):
        for i in range(H - K + 1):
            segment = [grid[i+k][j] for k in range(K)]
            if 'x' in segment:
                continue
            count = segment.count('.')
            if count < min_ops:
                min_ops = count
    
    if min_ops != float('inf'):
        print(min_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()