n, k = map(int, input().split())
a = list(map(int, input().split()))
s = set(a)
sum_total = k * (k + 1) // 2
sum_a = sum(x for x in s if x <= k)
print(sum_total - sum_a)