# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
L = int(data[1])
R = int(data[2])
A = list(map(int, data[3:]))

result = []

for i in range(N):
    if A[i] < L:
        result.append(L)
    elif A[i] > R:
        result.append(R)
    else:
        result.append(A[i])

print(" ".join(map(str, result)))