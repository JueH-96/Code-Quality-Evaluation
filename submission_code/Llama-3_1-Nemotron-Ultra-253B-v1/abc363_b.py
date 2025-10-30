n, t, p = map(int, input().split())
l = list(map(int, input().split()))
d = [max(0, t - li) for li in l]
d_sorted = sorted(d)
initial_count = sum(1 for x in d if x == 0)
print(0 if initial_count >= p else d_sorted[p-1])