# YOUR CODE HERE
def min_operations_to_consecutive_o(H, W, K, grid):
    def check_horizontal():
        min_ops = float('inf')
        for i in range(H):
            for j in range(W - K + 1):
                count_o = 0
                count_dot = 0
                for l in range(K):
                    if grid[i][j + l] == 'o':
                        count_o += 1
                    elif grid[i][j + l] == '.':
                        count_dot += 1
                if count_o + count_dot == K:
                    min_ops = min(min_ops, count_dot)
        return min_ops

    def check_vertical():
        min_ops = float('inf')
        for j in range(W):
            for i in range(H - K + 1):
                count_o = 0
                count_dot = 0
                for l in range(K):
                    if grid[i + l][j] == 'o':
                        count_o += 1
                    elif grid[i + l][j] == '.':
                        count_dot += 1
                if count_o + count_dot == K:
                    min_ops = min(min_ops, count_dot)
        return min_ops

    min_ops_horizontal = check_horizontal()
    min_ops_vertical = check_vertical()
    
    min_ops = min(min_ops_horizontal, min_ops_vertical)
    
    if min_ops == float('inf'):
        return -1
    else:
        return min_ops

import sys
input = sys.stdin.read
data = input().split()
H = int(data[0])
W = int(data[1])
K = int(data[2])
grid = data[3:]

print(min_operations_to_consecutive_o(H, W, K, grid))