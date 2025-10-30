n, s, m, l = map(int, input().split())
min_cost = float('inf')

for x in range(101):
    for y in range(101):
        for z in range(101):
            total_eggs = 6 * x + 8 * y + 12 * z
            if total_eggs >= n:
                cost = s * x + m * y + l * z
                if cost < min_cost:
                    min_cost = cost

print(min_cost)