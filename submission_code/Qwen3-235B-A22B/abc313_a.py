n = int(input())
p = list(map(int, input().split()))
if n == 1:
    print(0)
else:
    p0 = p[0]
    others = p[1:]
    max_other = max(others)
    if p0 > max_other:
        print(0)
    else:
        print(max_other - p0 + 1)