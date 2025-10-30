import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
M = int(data[2])
L = int(data[3])

min_cost = float('inf')

# Try all possible combinations of packs
for i in range(N // 12 + 1):
    for j in range(N // 8 + 1):
        for k in range(N // 6 + 1):
            total_eggs = 12 * i + 8 * j + 6 * k
            if total_eggs >= N:
                cost = 1200 * i + 150 * j + 120 * k
                min_cost = min(min_cost, cost)

print(min_cost)