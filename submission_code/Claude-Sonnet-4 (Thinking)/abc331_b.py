N, S, M, L = map(int, input().split())

min_cost = float('inf')

max_x = (N + 5) // 6
max_y = (N + 7) // 8
max_z = (N + 11) // 12

for x in range(max_x + 1):
    for y in range(max_y + 1):
        for z in range(max_z + 1):
            eggs = 6 * x + 8 * y + 12 * z
            if eggs >= N:
                cost = S * x + M * y + L * z
                min_cost = min(min_cost, cost)

print(min_cost)