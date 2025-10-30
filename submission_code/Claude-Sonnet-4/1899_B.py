t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    max_diff = 0
    
    # Try all possible values of k
    for k in range(1, n + 1):
        if n % k != 0:  # k must divide n
            continue
            
        num_trucks = n // k
        truck_weights = []
        
        # Calculate weight of each truck
        for i in range(num_trucks):
            weight = sum(a[i * k : (i + 1) * k])
            truck_weights.append(weight)
        
        # Find max difference between trucks
        if len(truck_weights) > 1:
            diff = max(truck_weights) - min(truck_weights)
            max_diff = max(max_diff, diff)
    
    print(max_diff)