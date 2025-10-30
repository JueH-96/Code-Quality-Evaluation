# YOUR CODE HERE
N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Sort fares in descending order to prioritize using passes on expensive days
F_sorted = sorted(F, reverse=True)

min_cost = float('inf')

# Try all possible numbers of pass batches
max_batches = (N + D - 1) // D  # This is ceil(N/D)
for k in range(max_batches + 1):
    # Cost of buying k batches of passes
    pass_cost = k * P
    
    # Number of passes we have
    num_passes = k * D
    
    # Use passes on the most expensive days (up to num_passes days)
    # Pay regular fare for the remaining days
    regular_fare_cost = sum(F_sorted[num_passes:])
    
    total_cost = pass_cost + regular_fare_cost
    min_cost = min(min_cost, total_cost)

print(min_cost)