N, M = map(int, input().split())

if N == 1:
    total = M + 1
else:
    total = 0
    for i in range(M + 1):
        term = N ** i
        total += term
        if total > 10**9:
            break

if total > 10**9:
    print("inf")
else:
    print(total)