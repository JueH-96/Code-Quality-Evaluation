# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            truck_weights = []
            for i in range(n // k):
                truck_weight = sum(a[i * k:(i + 1) * k])
                truck_weights.append(truck_weight)
            
            if len(truck_weights) > 1:
                max_diff = max(max_diff, max(truck_weights) - min(truck_weights))

    print(max_diff)


t = int(input())
for _ in range(t):
    solve()