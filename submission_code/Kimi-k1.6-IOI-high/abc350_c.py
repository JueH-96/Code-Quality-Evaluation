n = int(input())
A = list(map(int, input().split()))
swaps = []

for i in range(n):
    while A[i] != i + 1:
        x = A[i]
        A[i], A[x-1] = A[x-1], A[i]
        swaps.append((i+1, x))

print(len(swaps))
for a, b in swaps:
    print(a, b)