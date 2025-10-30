# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

C = sorted(A + B)

for i in range(1, N+M):
    if C[i] == C[i-1] + 1:
        print("Yes")
        break
else:
    print("No")