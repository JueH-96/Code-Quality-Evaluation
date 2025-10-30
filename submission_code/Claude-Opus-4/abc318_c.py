# YOUR CODE HERE
N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Sort fares in descending order to use passes on most expensive days
F.sort(reverse=True)

min_cost = float('inf')

# Try different numbers of passes
# We need at least ceil(k/D) batches to get k passes
for k in range(N + 1):
    # Number of batches needed to get at least k passes
    batches = (k + D - 1) // D if k > 0 else 0
    
    # Cost of passes
    pass_cost = batches * P
    
    # Cost of regular fares for days not covered by passes
    regular_cost = sum(F[k:])
    
    # Total cost
    total_cost = pass_cost + regular_cost
    
    min_cost = min(min_cost, total_cost)

print(min_cost)