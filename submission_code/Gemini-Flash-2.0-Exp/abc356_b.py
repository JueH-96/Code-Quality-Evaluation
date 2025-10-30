N, M = map(int, input().split())
A = list(map(int, input().split()))
X = []
for _ in range(N):
    X.append(list(map(int, input().split())))

nutrient_totals = [0] * M
for i in range(N):
    for j in range(M):
        nutrient_totals[j] += X[i][j]

met_goals = True
for i in range(M):
    if nutrient_totals[i] < A[i]:
        met_goals = False
        break

if met_goals:
    print("Yes")
else:
    print("No")