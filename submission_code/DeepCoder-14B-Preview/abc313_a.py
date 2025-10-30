n = int(input())
p = list(map(int, input().split()))
if n == 1:
    print(0)
else:
    p1 = p[0]
    max_rest = max(p[1:])
    x = max(0, max_rest - p1 + 1)
    print(x)