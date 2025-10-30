# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

D = int(data[0])

x = 0
y = 0
min_diff = D

for i in range(int(D**0.5) + 1):
    for j in range(int(D**0.5) + 1):
        diff = abs(i**2 + j**2 - D)
        if diff < min_diff:
            min_diff = diff

print(min_diff)