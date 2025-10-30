# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
A = list(map(int, input[2:2+N]))
B = list(map(int, input[2+N:]))

C = sorted(A + B)
A_set = set(A)

for i in range(1, len(C)):
    if C[i] in A_set and C[i-1] in A_set:
        print("Yes")
        break
else:
    print("No")