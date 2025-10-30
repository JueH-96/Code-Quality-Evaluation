def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_diff = 0
    
    # Try all possible k values (divisors of n)
    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            
            # Calculate weight of each truck
            truck_weights = []
            for i in range(num_trucks):
                truck_weight = sum(a[i * k:(i + 1) * k])
                truck_weights.append(truck_weight)
            
            # Find max difference
            if len(truck_weights) > 1:
                max_weight = max(truck_weights)
                min_weight = min(truck_weights)
                max_diff = max(max_diff, max_weight - min_weight)
    
    return max_diff

t = int(input())
for _ in range(t):
    print(solve())