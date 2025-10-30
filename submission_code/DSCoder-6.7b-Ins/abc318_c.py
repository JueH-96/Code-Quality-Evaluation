# YOUR CODE HERE

import sys

N, D, P = map(int, sys.stdin.readline().split())
F = list(map(int, sys.stdin.readline().split()))

F.sort()

total_cost = 0

# Calculate the total cost for the days covered by one-day passes
total_cost += (N // D) * P

# Calculate the total cost for the leftover days
total_cost += sum(F[:N % D])

print(total_cost)