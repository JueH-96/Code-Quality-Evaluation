def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_difference_overall = 0
    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            truck_weights = []
            for j in range(num_trucks):
                current_truck_weight = 0
                for i in range(k):
                    current_truck_weight += a[j * k + i]
                truck_weights.append(current_truck_weight)
            if num_trucks > 1:
                min_weight = min(truck_weights)
                max_weight = max(truck_weights)
                difference = max_weight - min_weight
            else:
                difference = 0
            max_difference_overall = max(max_difference_overall, difference)
    print(max_difference_overall)

t = int(input())
for _ in range(t):
    solve()