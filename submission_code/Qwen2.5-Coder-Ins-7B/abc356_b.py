# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
X = []

for i in range(N):
    X.append(list(map(int, data[2+M+i*M:2+M+(i+1)*M])))

met_goal = True

for j in range(M):
    total = sum(X[i][j] for i in range(N))
    if total < A[j]:
        met_goal = False
        break

if met_goal:
    print("Yes")
else:
    print("No")