import math

# Read input
N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Sort fares in descending order
F.sort(reverse=True)

min_cost = float('inf')

# Try different numbers of batches
max_batches = math.ceil(N / D)
for k in range(max_batches + 1):
    passes = k * D
    cost = k * P
    
    # Add regular fares for days not covered by passes
    if passes < N:
        cost += sum(F[passes:])
    
    min_cost = min(min_cost, cost)

print(min_cost)