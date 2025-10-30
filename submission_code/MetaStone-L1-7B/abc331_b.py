N, S, M, L = map(int, input().split())

min_cost = float('inf')
max_a = (N + 11) // 12 + 2
max_b = (N + 7) // 8 + 2
max_c = (N) // 6 + 2

for a in range(max_a + 1):
    for b in range(max_b + 1):
        for c in range(max_c + 1):
            total_eggs = 12 * a + 8 * b + 6 * c
            if total_eggs >= N:
                cost = a * L + b * M + c * S
                if cost < min_cost:
                    min_cost = cost

print(min_cost)