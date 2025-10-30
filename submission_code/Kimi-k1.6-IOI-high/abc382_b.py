n, d = map(int, input().split())
s = list(input().strip())

for _ in range(d):
    for i in reversed(range(n)):
        if s[i] == '@':
            s[i] = '.'
            break

print(''.join(s))