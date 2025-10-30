# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

A = list(map(int, data[index:index + M]))
index += M

X = []
for i in range(N):
    X.append(list(map(int, data[index:index + M])))
    index += M

nutrient_totals = [0] * M
for i in range(N):
    for j in range(M):
        nutrient_totals[j] += X[i][j]

met_goals = all(nutrient_totals[j] >= A[j] for j in range(M))

if met_goals:
    print("Yes")
else:
    print("No")