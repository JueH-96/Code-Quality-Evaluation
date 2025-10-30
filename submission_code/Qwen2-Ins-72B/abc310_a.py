N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

min_D = min(D)
min_total = min(P, Q + min_D)

print(min_total)