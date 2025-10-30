n, t, p = map(int, input().split())
l = list(map(int, input().split()))
d = [max(0, t - x) for x in l]
d.sort()
print(d[p-1])