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

total_price = 0

for a in A:
    for b in B:
        total_price += min(a + b, P)

print(total_price)