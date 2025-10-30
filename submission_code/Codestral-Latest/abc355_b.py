# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:]))

C = sorted(A + B)

for i in range(len(C) - 1):
    if C[i] in A and C[i+1] in A:
        print("Yes")
        break
else:
    print("No")