def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = 0
    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            if num_trucks == 1:
                ans = max(ans, 0)
            else:
                truck_weights = []
                for i in range(num_trucks):
                    truck_weight = 0
                    for j in range(k):
                        truck_weight += a[i * k + j]
                    truck_weights.append(truck_weight)
                
                ans = max(ans, max(truck_weights) - min(truck_weights))
    print(ans)

t = int(input())
for _ in range(t):
    solve()