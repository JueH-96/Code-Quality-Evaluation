def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_abs_diff = 0

    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            if num_trucks == 1:
                continue

            truck_weights = [0] * num_trucks
            for i in range(n):
                truck_index = i // k
                truck_weights[truck_index] += a[i]

            current_abs_diff = max(truck_weights) - min(truck_weights)
            max_abs_diff = max(max_abs_diff, current_abs_diff)

    print(max_abs_diff)

t = int(input())
for _ in range(t):
    solve()