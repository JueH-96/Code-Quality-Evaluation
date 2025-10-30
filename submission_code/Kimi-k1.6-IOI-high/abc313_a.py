n = int(input())
p = list(map(int, input().split()))
if n == 1:
    print(0)
else:
    max_others = max(p[1:])
    required = max_others - p[0] + 1
    print(max(required, 0))