n = int(input().strip())
for i in range(n):
    line = []
    for j in range(n):
        d = min(i + 1, j + 1, n - i, n - j)
        if d % 2 == 1:
            line.append('#')
        else:
            line.append('.')
    print(''.join(line))