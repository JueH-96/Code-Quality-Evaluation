N, D, P = map(int, input().split())
F = list(map(int, input().split()))
F.sort(reverse=True)

pre_sum = [0] * (N + 1)
for i in range(1, N + 1):
    pre_sum[i] = pre_sum[i - 1] + F[i - 1]

total_sum = pre_sum[N]
K_min = (N + D - 1) // D
min_cost = float('inf')

for K in range(K_min + 1):
    used = K * D
    if used <= N:
        sum_top = pre_sum[used]
    else:
        sum_top = pre_sum[N]
    cost = K * P + (total_sum - sum_top)
    if cost < min_cost:
        min_cost = cost

print(min_cost)