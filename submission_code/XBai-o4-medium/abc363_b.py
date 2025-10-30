n, t, p = map(int, input().split())
l = list(map(int, input().split()))
days = [max(0, t - x) for x in l]
days.sort()
print(days[p-1])