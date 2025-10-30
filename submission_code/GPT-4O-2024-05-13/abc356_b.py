# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+M]))
X = []

index = 2 + M
for i in range(N):
    X.append(list(map(int, data[index:index+M])))
    index += M

total_nutrients = [0] * M

for i in range(N):
    for j in range(M):
        total_nutrients[j] += X[i][j]

goal_met = True
for j in range(M):
    if total_nutrients[j] < A[j]:
        goal_met = False
        break

if goal_met:
    print("Yes")
else:
    print("No")