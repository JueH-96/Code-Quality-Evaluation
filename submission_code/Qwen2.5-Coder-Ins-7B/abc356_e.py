# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

result = 0
for i in range(N-1):
    for j in range(i+1, N):
        result += (max(A[i], A[j]) // min(A[i], A[j]))

print(result)