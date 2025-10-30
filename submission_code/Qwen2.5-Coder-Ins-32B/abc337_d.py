# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    K = int(input[2])
    grid = input[3:]

    def min_operations_to_fill_line(line, K):
        n = len(line)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if line[i] == 'x':
                dp[i + 1] = float('inf')
            else:
                dp[i + 1] = dp[i]
            if i >= K - 1:
                if line[i - K + 1] != 'x':
                    dp[i + 1] = min(dp[i + 1], dp[i - K + 1] + (K - line[i - K + 1:i + 1].count('o')))
        return min(dp[n], float('inf'))

    min_operations = float('inf')
    for i in range(H):
        min_operations = min(min_operations, min_operations_to_fill_line(grid[i], K))

    for j in range(W):
        column = ''.join(grid[i][j] for i in range(H))
        min_operations = min(min_operations, min_operations_to_fill_line(column, K))

    if min_operations == float('inf'):
        print(-1)
    else:
        print(min_operations)

if __name__ == "__main__":
    main()