# YOUR CODE HERE
n = int(input())
s = [input() for _ in range(n)]
m = max(len(si) for si in s)
t = [['*' for _ in range(n)] for _ in range(m)]

for i in range(n):
    for j in range(len(s[i])):
        t[j][n - 1 - i] = s[i][j]

for i in range(m):
    row = ""
    for j in range(n):
        row += t[i][j]
    while row.endswith("*"):
        row = row[:-1]
    print(row)