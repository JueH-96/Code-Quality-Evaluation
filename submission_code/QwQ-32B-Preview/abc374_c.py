N = int(input())
K = list(map(int, input().split()))

total_sum = sum(K)
min_max_sum = float('inf')

for subset in range(1, (1 << N)):
    subset_sum = 0
    for i in range(N):
        if (subset & (1 << i)):
            subset_sum += K[i]
    complement_sum = total_sum - subset_sum
    max_sum = max(subset_sum, complement_sum)
    if max_sum < min_max_sum:
        min_max_sum = max_sum

print(min_max_sum)