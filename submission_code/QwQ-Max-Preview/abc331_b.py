n, s, m, l = map(int, input().split())

max_a = (n + 12) // 6
max_b = (n + 12) // 8
max_c = (n + 12) // 12

min_cost = float('inf')

for a in range(max_a + 1):
    for b in range(max_b + 1):
        for c in range(max_c + 1):
            total_eggs = 6 * a + 8 * b + 12 * c
            if total_eggs >= n:
                cost = a * s + b * m + c * l
                if cost < min_cost:
                    min_cost = cost

print(min_cost)