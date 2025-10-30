n = int(input())
cards = []
for i in range(n):
    a, c = map(int, input().split())
    cards.append((a, c, i + 1))

# Sort by strength in descending order
cards.sort(key=lambda x: x[0], reverse=True)

pareto_optimal = []
min_cost = float('inf')

for a, c, idx in cards:
    if c <= min_cost:
        pareto_optimal.append(idx)
        min_cost = c

# Sort the result by card index
pareto_optimal.sort()

print(len(pareto_optimal))
print(' '.join(map(str, pareto_optimal)))