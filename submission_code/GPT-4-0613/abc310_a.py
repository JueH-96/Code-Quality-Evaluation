N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

min_dish = min(D)
if min_dish + Q < P:
    print(min_dish + Q)
else:
    print(P)