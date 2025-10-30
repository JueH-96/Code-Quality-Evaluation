N, S, M, L = map(int, input().split())
min_cost = float('inf')

max_a = (N // 6) + 2
max_b = (N // 8) + 2
max_c = (N // 12) + 2

for a in range(max_a + 1):
    for b in range(max_b + 1):
        for c in range(max_c + 1):
            total_eggs = 6 * a + 8 * b + 12 * c
            if total_eggs >= N:
                cost = a * S + b * M + c * L
                if cost < min_cost:
                    min_cost = cost

print(min_cost)