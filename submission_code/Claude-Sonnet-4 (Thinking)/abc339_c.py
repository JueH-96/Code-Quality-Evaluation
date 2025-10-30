n = int(input())
a = list(map(int, input().split()))

prefix_sum = 0
min_prefix = 0

for i in range(n):
    prefix_sum += a[i]
    min_prefix = min(min_prefix, prefix_sum)

x_min = max(0, -min_prefix)
current = x_min + prefix_sum

print(current)