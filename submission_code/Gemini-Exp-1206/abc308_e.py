def mex(a, b, c):
    for i in range(4):
        if i != a and i != b and i != c:
            return i

n = int(input())
a = list(map(int, input().split()))
s = input()

ans = 0
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
                ans += mex(a[i], a[j], a[k])

print(ans)