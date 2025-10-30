n = int(input())
res = []
for i in range(1, n + 1):
    if i % 3 == 0:
        res.append('x')
    else:
        res.append('o')
print(''.join(res))