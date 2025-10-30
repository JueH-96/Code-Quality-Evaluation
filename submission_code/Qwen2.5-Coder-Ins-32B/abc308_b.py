# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
C = input[2:2+N]
D = input[2+N:2+N+M]
P = list(map(int, input[2+N+M:]))

price_map = {d: p for d, p in zip(D, P[1:])}
total_price = sum(price_map.get(c, P[0]) for c in C)

print(total_price)