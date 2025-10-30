n, k = map(int, input().split())
a = list(map(int, input().split()))
unique_a = set(a)
sum_present = sum(x for x in unique_a if x <= k)
total_sum = k * (k + 1) // 2
print(total_sum - sum_present)