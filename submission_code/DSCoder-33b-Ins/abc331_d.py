# YOUR CODE HERE

N, Q = map(int, input().split())
grid = [input() for _ in range(N)]

# Create a prefix sum grid
prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(N):
    for j in range(N):
        prefix_sum[i + 1][j + 1] = prefix_sum[i + 1][j] + prefix_sum[i][j + 1] - prefix_sum[i][j] + (grid[i][j] == 'B')

# Process queries
for _ in range(Q):
    A, B, C, D = map(int, input().split())
    A %= N
    B %= N
    C %= N
    D %= N
    if A > C or B > D:
        A, C = C, A
        B, D = D, B
    count = prefix_sum[C + 1][D + 1] - prefix_sum[A][D + 1] - prefix_sum[C + 1][B] + prefix_sum[A][B]
    print(count)