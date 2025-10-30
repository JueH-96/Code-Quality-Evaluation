import math

N = int(input())

total_cost = 0

while N >= 2:
    total_cost += N
    N = math.floor(N / 2) + math.ceil(N / 2)

print(total_cost)