t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    max_diff = 0
    
    # Try all possible k values where n % k == 0
    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            if num_trucks == 1:
                continue  # Only one truck, difference is 0
            
            max_weight = float('-inf')
            min_weight = float('inf')
            
            # Calculate weight for each truck
            for truck in range(num_trucks):
                start_idx = truck * k
                truck_weight = sum(a[start_idx:start_idx + k])
                max_weight = max(max_weight, truck_weight)
                min_weight = min(min_weight, truck_weight)
            
            diff = max_weight - min_weight
            max_diff = max(max_diff, diff)
    
    print(max_diff)