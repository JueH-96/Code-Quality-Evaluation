n, d = map(int, input().split())
s = list(input().strip())

for _ in range(d):
    for i in range(n-1, -1, -1):
        if s[i] == '@':
            s[i] = '.'
            break

print(''.join(s))