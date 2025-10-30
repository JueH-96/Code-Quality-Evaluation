def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    for k in range(1, n + 1):
        if n % k == 0:
            max_diff = 0
            num_trucks = n // k
            if num_trucks == 1:
                print(0)
                continue
            
            truck_weights = []
            for i in range(num_trucks):
                truck_weight = 0
                for j in range(k):
                    truck_weight += a[i * k + j]
                truck_weights.append(truck_weight)
            
            max_weight = max(truck_weights)
            min_weight = min(truck_weights)
            max_diff = max_weight - min_weight
            print(max_diff)
            break
    

t = int(input())
for _ in range(t):
    solve()