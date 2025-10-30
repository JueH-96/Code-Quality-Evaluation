def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    divisors = []
    for k in range(1, n + 1):
        if n % k == 0:
            divisors.append(k)
            
    max_diff_overall = 0
    
    for k in divisors:
        num_trucks = n // k
        truck_weights = []
        for i in range(num_trucks):
            current_truck_weight = 0
            for j in range(k):
                current_truck_weight += a[i * k + j]
            truck_weights.append(current_truck_weight)
            
        if not truck_weights:
            diff = 0
        else:
            max_weight = max(truck_weights)
            min_weight = min(truck_weights)
            diff = max_weight - min_weight
            
        max_diff_overall = max(max_diff_overall, diff)
        
    print(max_diff_overall)

t = int(input())
for _ in range(t):
    solve()