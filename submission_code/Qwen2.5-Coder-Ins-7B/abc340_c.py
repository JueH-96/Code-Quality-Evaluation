# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])

total_cost = 0

while N >= 2:
    if N % 2 == 0:
        total_cost += N
        N = N // 2
    else:
        total_cost += N
        N = (N // 2) + 1

print(total_cost)