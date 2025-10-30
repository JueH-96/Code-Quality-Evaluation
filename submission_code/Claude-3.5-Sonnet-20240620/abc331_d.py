# YOUR CODE HERE
def count_black_squares(grid, N, A, B, C, D):
    def count_in_range(x1, y1, x2, y2):
        count = 0
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if grid[i % N][j % N] == 'B':
                    count += 1
        return count

    return count_in_range(A, B, C, D)

N, Q = map(int, input().split())
grid = [input().strip() for _ in range(N)]

for _ in range(Q):
    A, B, C, D = map(int, input().split())
    result = count_black_squares(grid, N, A, B, C, D)
    print(result)