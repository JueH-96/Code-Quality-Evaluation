n = int(input())
strs = [input() for _ in range(n)]
max_len = max(len(s) for s in strs)
res = [['*'] * n for _ in range(max_len)]
for i, s in enumerate(strs):
    for j, c in enumerate(s):
        res[j][i] = c

for row in res:
    while row[-1] == '*':
        row.pop()
    print(''.join(row))