import math

N, S, M, L = map(int, input().split())

min_cost = float('inf')

# Upper bounds: maximum number of each pack type we would need
max_x = math.ceil(N / 6)
max_y = math.ceil(N / 8)
max_z = math.ceil(N / 12)

for x in range(max_x + 1):
    for y in range(max_y + 1):
        for z in range(max_z + 1):
            if 6*x + 8*y + 12*z >= N:
                cost = S*x + M*y + L*z
                min_cost = min(min_cost, cost)

print(min_cost)