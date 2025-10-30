# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Compute row and column ranks
row_rank = [0] * (N + 1)
col_rank = [0] * (N + 1)

for i in range(N):
    row_rank[P[i]] = i + 1
    col_rank[Q[i]] = i + 1

# Construct and print the grid
for i in range(1, N + 1):
    row = []
    for j in range(1, N + 1):
        if (row_rank[i] - 1) + (col_rank[j] - 1) >= N - 1:
            row.append('1')
        else:
            row.append('0')
    print(''.join(row))