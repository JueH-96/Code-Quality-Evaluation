# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
q = list(map(int, input().split()))

grid = [[0 for _ in range(n)] for _ in range(n)]

p_inv = [0] * (n + 1)
for i in range(n):
    p_inv[p[i]] = i

q_inv = [0] * (n + 1)
for i in range(n):
    q_inv[q[i]] = i

for i in range(n):
    for j in range(n):
        if p_inv[i+1] < p_inv[j+1]:
            grid[i][j] = 0
        else:
            grid[i][j] = 1

ans = []
for row in grid:
    ans.append("".join(map(str, row)))

for row in ans:
    print(row)