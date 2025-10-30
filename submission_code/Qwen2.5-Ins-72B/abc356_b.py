# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:M+2]))
X = [list(map(int, data[M+2+i*M:M+2+(i+1)*M])) for i in range(N)]

nutrient_sums = [0] * M

for i in range(N):
    for j in range(M):
        nutrient_sums[j] += X[i][j]

for i in range(M):
    if nutrient_sums[i] < A[i]:
        print("No")
        sys.exit(0)

print("Yes")