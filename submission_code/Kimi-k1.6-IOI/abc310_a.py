n, p, q = map(int, input().split())
d = list(map(int, input().split()))
min_d = min(d)
print(min(p, q + min_d))