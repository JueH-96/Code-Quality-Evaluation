# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
P = int(data[2])
A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:]))

A.sort()
B.sort()

total = 0
j = M - 1
for a in A:
    while j >= 0 and a + B[j] > P:
        j -= 1
    if j == -1:
        total += P * M
        break
    total += (j + 1) * P + (M - j - 1) * (a + B[0])

print(total)