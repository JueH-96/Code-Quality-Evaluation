n, t, p = map(int, input().split())
l = list(map(int, input().split()))
required = [max(0, t - x) for x in l]
count_zero = required.count(0)
if count_zero >= p:
    print(0)
else:
    required.sort()
    print(required[p-1])