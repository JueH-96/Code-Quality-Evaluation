import sys

def min_operations(H, W, K, grid):
    min_ops = float('inf')

    # Check horizontal sequences
    for i in range(H):
        for j in range(W - K + 1):
            ops = 0
            for k in range(K):
                if grid[i][j + k] == '.':
                    ops += 1
                elif grid[i][j + k] == 'x':
                    break
            else:
                min_ops = min(min_ops, ops)

    # Check vertical sequences
    for i in range(H - K + 1):
        for j in range(W):
            ops = 0
            for k in range(K):
                if grid[i + k][j] == '.':
                    ops += 1
                elif grid[i + k][j] == 'x':
                    break
            else:
                min_ops = min(min_ops, ops)

    return min_ops if min_ops != float('inf') else -1

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    result = min_operations(H, W, K, grid)
    print(result)

if __name__ == "__main__":
    main()