def solve():
    n, k = map(int, input().split())
    s = list(input())
    operations = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            operations += 1
            for j in range(i, min(n, i + k)):
                s[j] = 'W'
            i += k
        else:
            i += 1
    print(operations)

t = int(input())
for _ in range(t):
    solve()