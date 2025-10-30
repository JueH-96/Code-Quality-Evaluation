n, *rest = map(int, open(0).read().split())
a = rest[:n]

sum_so_far = 0
min_sum = float('inf')
for num in a:
    sum_so_far += num
    if sum_so_far < min_sum:
        min_sum = sum_so_far

x = max(0, -min_sum)
current = x + sum_so_far
print(current)