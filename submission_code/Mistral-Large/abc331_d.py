import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

grid = []
index = 2
for _ in range(N):
    grid.append(data[index])
    index += 1

queries = []
for _ in range(Q):
    A, B, C, D = map(int, data[index:index+4])
    queries.append((A, B, C, D))
    index += 4

def count_black_squares(A, B, C, D, N, grid):
    count = 0
    for i in range(A, C + 1):
        for j in range(B, D + 1):
            if grid[i % N][j % N] == 'B':
                count += 1
    return count

results = []
for query in queries:
    A, B, C, D = query
    results.append(count_black_squares(A, B, C, D, N, grid))

sys.stdout.write('
'.join(map(str, results)) + '
')