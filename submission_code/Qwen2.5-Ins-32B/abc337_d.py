import sys

def min_operations_to_consecutive_o(H, W, K, grid):
    def min_ops_horizontal():
        min_ops = float('inf')
        for row in grid:
            count = 0
            for i in range(W - K + 1):
                count += row[i:i+K].count('.')
                if i > 0:
                    count -= row[i-1] == '.'
                if i + K < W:
                    count -= row[i+K] == '.'
                min_ops = min(min_ops, count)
        return min_ops

    def min_ops_vertical():
        min_ops = float('inf')
        for col in range(W):
            count = 0
            for i in range(H - K + 1):
                count += sum(grid[j][col] == '.' for j in range(i, i + K))
                if i > 0:
                    count -= grid[i-1][col] == '.'
                if i + K < H:
                    count -= grid[i+K][col] == '.'
                min_ops = min(min_ops, count)
        return min_ops

    horizontal_ops = min_ops_horizontal()
    vertical_ops = min_ops_vertical()
    result = min(horizontal_ops, vertical_ops)
    return result if result != float('inf') else -1

if __name__ == "__main__":
    H, W, K = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    print(min_operations_to_consecutive_o(H, W, K, grid))