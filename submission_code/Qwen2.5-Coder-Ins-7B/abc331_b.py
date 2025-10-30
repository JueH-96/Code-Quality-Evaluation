# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
M = int(data[2])
L = int(data[3])

min_cost = float('inf')

for i in range(N // 12 + 1):
    for j in range(N // 8 + 1):
        for k in range(N // 6 + 1):
            if 12 * i + 8 * j + 6 * k >= N:
                min_cost = min(min_cost, L * i + M * j + S * k)

print(min_cost)