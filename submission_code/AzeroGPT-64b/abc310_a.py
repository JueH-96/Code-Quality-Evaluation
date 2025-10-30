N, P, Q = map(int, input().split())
D = list(map(int, input().split()))
min_price = min(D)
if P <= Q + min_price:
    print(P)
else:
    print(Q + min_price)