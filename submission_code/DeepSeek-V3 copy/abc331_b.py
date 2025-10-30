# YOUR CODE HERE
N, S, M, L = map(int, input().split())
min_cost = float('inf')

# Determine the maximum number of each pack needed
max_6 = (N // 6) + 2
max_8 = (N // 8) + 2
max_12 = (N // 12) + 2

for a in range(max_6):
    for b in range(max_8):
        for c in range(max_12):
            total_eggs = 6*a + 8*b + 12*c
            if total_eggs >= N:
                total_cost = S*a + M*b + L*c
                if total_cost < min_cost:
                    min_cost = total_cost

print(min_cost)