n = int(input())
strings = [input().strip() for _ in range(n)]
max_len = max(len(s) for s in strings)
padded = [s.ljust(max_len, '*') for s in strings]

for j in range(max_len):
    column = []
    for s in reversed(padded):
        column.append(s[j])
    t = ''.join(column)
    t = t.rstrip('*')
    print(t)