# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = int(data[1])
Q = int(data[2])
D = list(map(int, data[3:]))

# If the discount is not used, the cost is P
min_cost = P

# If the discount is used, we need to find the minimum cost of a dish
# and add the discount price to it
for d in D:
    min_cost = min(min_cost, Q + d)

print(min_cost)