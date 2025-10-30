# YOUR CODE HERE
n, m = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(n)]

met_goals = True
for i in range(m):
    if sum(X[j][i] for j in range(n)) < A[i]:
        met_goals = False
        break

print("Yes" if met_goals else "No")