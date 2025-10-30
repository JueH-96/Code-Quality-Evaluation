n, k = map(int, input().split())
a = list(map(int, input().split()))
s = set(a)
sum_in = sum(x for x in s if x <= k)
total = k * (k + 1) // 2
print(total - sum_in)