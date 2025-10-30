n, t, p = map(int, input().split())
l = list(map(int, input().split()))

current = sum(1 for x in l if x >= t)
if current >= p:
    print(0)
else:
    needed = p - current
    days = [t - x for x in l if x < t]
    days.sort()
    print(days[needed - 1])