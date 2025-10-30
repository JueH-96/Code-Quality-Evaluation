n = int(input())
p = list(map(int, input().split()))

if n == 1:
    print(0)
else:
    current = p[0]
    max_other = max(p[1:])
    if current > max_other:
        print(0)
    else:
        print(max_other - current + 1)