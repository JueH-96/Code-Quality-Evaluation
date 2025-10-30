# YOUR CODE HERE
n = int(input())
s = [input() for _ in range(n)]
m = max(len(si) for si in s)
t = [''] * m
for j in range(m):
    for i in range(n):
        if j < len(s[i]):
            if len(t[j]) < n - i:
                t[j] += s[i][j]
            else:
                t[j] = t[j][:n-i-1] + s[i][j]
        else:
            if len(t[j]) < n - i:
                t[j] += '*'
for tj in t:
    while tj and tj[-1] == '*':
        tj = tj[:-1]
    print(tj.ljust(n, '*'))