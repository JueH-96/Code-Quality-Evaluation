n, k = map(int, input().split())
a = list(map(int, input().split()))
unique_valid = {x for x in a if x <= k}
sum_unique = sum(unique_valid)
total = k * (k + 1) // 2 - sum_unique
print(total)