N = int(input())
A = list(map(int, input().split()))
B = sorted([(a, i+1) for i, a in enumerate(A)])
swaps = []
for i in range(N):
    while B[i][1] != i+1:
        swaps.append((B[i][1], B[B[i][1]-1][1]))
        B[B[i][1]-1], B[i] = B[i], B[B[i][1]-1]
print(len(swaps))
for swap in swaps:
    print(*swap)