n, d = map(int, input().split())
s = list(input().strip())

for _ in range(d):
    pos = -1
    # Find the rightmost '@'
    for i in range(n-1, -1, -1):
        if s[i] == '@':
            pos = i
            break
    # Replace that position with '.'
    s[pos] = '.'

print(''.join(s))