def max_weight_difference(n, weights):
    total_weight = sum(weights)
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i + 1] = prefix_sums[i] + weights[i]
    
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            continue
        trucks = n // k
        weight_per_truck = total_weight // (trucks + 1)
        for i in range(1, trucks + 2):
            left_weight = prefix_sums[min(i * k, n)]
            right_weight = prefix_sums[min((i - 1) * k, n)]
            max_diff = max(max_diff, abs(weight_per_truck - (left_weight - right_weight)))
    return max_diff

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    weights = list(map(int, input().strip().split()))
    print(max_weight_difference(n, weights))