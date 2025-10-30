N, M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]

nutrient_totals = [0] * M

for i in range(N):
    for j in range(M):
        nutrient_totals[j] += X[i][j]

for i in range(M):
    if nutrient_totals[i] < A[i]:
        print("No")
        exit()

print("Yes")