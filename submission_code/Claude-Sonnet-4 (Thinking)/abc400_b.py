N, M = map(int, input().split())

X = 0
power = 1  # N^0 = 1

for i in range(M + 1):
    X += power
    if X > 10**9:
        print("inf")
        break
    power *= N
else:
    print(X)