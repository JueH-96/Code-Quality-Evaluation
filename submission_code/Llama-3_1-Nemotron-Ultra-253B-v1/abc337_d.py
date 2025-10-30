import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    min_ops = float('inf')
    
    # Process horizontal windows
    for row in range(H):
        s = grid[row]
        x_prefix = [0] * (W + 1)
        dot_prefix = [0] * (W + 1)
        for i in range(W):
            x_prefix[i+1] = x_prefix[i] + (1 if s[i] == 'x' else 0)
            dot_prefix[i+1] = dot_prefix[i] + (1 if s[i] == '.' else 0)
        for j in range(W - K + 1):
            end = j + K
            if x_prefix[end] - x_prefix[j] == 0:
                cost = dot_prefix[end] - dot_prefix[j]
                if cost < min_ops:
                    min_ops = cost
    
    # Process vertical windows
    for col in range(W):
        x_prefix = [0] * (H + 1)
        dot_prefix = [0] * (H + 1)
        for i in range(H):
            cell = grid[i][col]
            x_prefix[i+1] = x_prefix[i] + (1 if cell == 'x' else 0)
            dot_prefix[i+1] = dot_prefix[i] + (1 if cell == '.' else 0)
        for i in range(H - K + 1):
            end = i + K
            if x_prefix[end] - x_prefix[i] == 0:
                cost = dot_prefix[end] - dot_prefix[i]
                if cost < min_ops:
                    min_ops = cost
    
    print(min_ops if min_ops != float('inf') else -1)

if __name__ == "__main__":
    main()