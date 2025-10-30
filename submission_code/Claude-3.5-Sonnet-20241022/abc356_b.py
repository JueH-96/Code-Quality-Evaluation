N, M = map(int, input().split())
goals = list(map(int, input().split()))
total = [0] * M

for _ in range(N):
    nutrients = list(map(int, input().split()))
    for j in range(M):
        total[j] += nutrients[j]

meets_goals = True
for i in range(M):
    if total[i] < goals[i]:
        meets_goals = False
        break

print("Yes" if meets_goals else "No")