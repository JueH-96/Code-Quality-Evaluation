N, M = map(int, input().split())

X = 0
power = 1
for i in range(M + 1):
    X += power
    if X > 10**9:
        print("inf")
        exit()
    power *= N

print(X)