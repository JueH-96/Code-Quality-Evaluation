n, d = map(int, input().split())
s = list(input())
for _ in range(d):
    # Find the rightmost '@'
    idx = len(s) - 1
    while idx >= 0 and s[idx] == '.':
        idx -= 1
    if idx >= 0:
        s.pop(idx)
print(''.join(s))