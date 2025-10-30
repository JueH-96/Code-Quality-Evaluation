def max_truck_weight_difference(n, weights):
    # Initialize result
    max_diff = 0
    
    # Precompute prefix sums for efficient segment sum calculation
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + weights[i]
    
    # Find all factors of n
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    
    # Check each factor
    for k in factors:
        num_trucks = n // k
        
        # If there's only one truck, the difference is 0
        if num_trucks == 1:
            continue
        
        # Find max and min truck weights
        max_weight = float('-inf')
        min_weight = float('inf')
        
        for i in range(num_trucks):
            # Calculate weight of the i-th truck using prefix sums
            truck_weight = prefix_sum[(i + 1) * k] - prefix_sum[i * k]
            max_weight = max(max_weight, truck_weight)
            min_weight = min(min_weight, truck_weight)
        
        # Update max difference
        diff = max_weight - min_weight
        max_diff = max(max_diff, diff)
    
    return max_diff

# Read number of test cases
t = int(input())
for _ in range(t):
    n = int(input())
    weights = list(map(int, input().split()))
    print(max_truck_weight_difference(n, weights))