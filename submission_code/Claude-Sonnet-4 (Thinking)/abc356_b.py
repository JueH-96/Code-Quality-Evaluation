N, M = map(int, input().split())
A = list(map(int, input().split()))

totals = [0] * M

for i in range(N):
    nutrients = list(map(int, input().split()))
    for j in range(M):
        totals[j] += nutrients[j]

if all(totals[j] >= A[j] for j in range(M)):
    print("Yes")
else:
    print("No")