n, k = map(int, input().split())
a = list(map(int, input().split()))
unique_a = set(a)
sum_in = sum(x for x in unique_a if x <= k)
total = k * (k + 1) // 2
print(total - sum_in)