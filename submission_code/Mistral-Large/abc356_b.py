import sys

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

A = [0] * M
for i in range(M):
    A[i] = int(data[index])
    index += 1

X = []
for i in range(N):
    row = [0] * M
    for j in range(M):
        row[j] = int(data[index])
        index += 1
    X.append(row)

total_nutrients = [0] * M
for i in range(N):
    for j in range(M):
        total_nutrients[j] += X[i][j]

meets_goal = all(total_nutrients[i] >= A[i] for i in range(M))

output = "Yes" if meets_goal else "No"
sys.stdout.write(output + "
")