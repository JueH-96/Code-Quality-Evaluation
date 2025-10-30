N = int(input())
A = list(map(int, input().split()))

total = 0
for i in range(N):
    seen = set()
    for j in range(i, N):
        seen.add(A[j])
        total += len(seen)

print(total)