n = int(input())
p = list(map(int, input().split()))
if n == 1:
    print(0)
else:
    p1 = p[0]
    max_other = max(p[1:])
    if p1 > max_other:
        print(0)
    else:
        print(max_other - p1 + 1)