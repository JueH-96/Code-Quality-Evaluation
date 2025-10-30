n, d = map(int, input().split())
s = list(input())
for i in range(d):
    rightmost = -1
    for j in range(n):
        if s[j] == '@':
            rightmost = j
    if rightmost != -1:
        s[rightmost] = '.'
print("".join(s))