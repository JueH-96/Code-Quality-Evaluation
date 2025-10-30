import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:]

    def count_changes(seq, K):
        n = len(seq)
        min_ops = float('inf')
        current_o = 0
        current_dots = 0

        for i in range(n):
            if seq[i] == 'o':
                current_o += 1
            elif seq[i] == '.':
                current_dots += 1

            if i >= K:
                if seq[i - K] == 'o':
                    current_o -= 1
                elif seq[i - K] == '.':
                    current_dots -= 1

            if i >= K - 1 and current_o + current_dots == K:
                min_ops = min(min_ops, current_dots)

        return min_ops if min_ops != float('inf') else -1

    min_ops = float('inf')

    # Check horizontal sequences
    for row in grid:
        result = count_changes(row, K)
        if result != -1:
            min_ops = min(min_ops, result)

    # Check vertical sequences
    for j in range(W):
        col = ''.join(grid[i][j] for i in range(H))
        result = count_changes(col, K)
        if result != -1:
            min_ops = min(min_ops, result)

    if min_ops == float('inf'):
        min_ops = -1

    print(min_ops)

if __name__ == "__main__":
    solve()