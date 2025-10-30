n = int(input())
a = list(map(int, input().split()))
prefix_sums = []
current = 0
for num in a:
    current += num
    prefix_sums.append(current)

max_neg = 0
for s in prefix_sums:
    current_max = -s
    if current_max > max_neg:
        max_neg = current_max

s_min = max(max_neg, 0)
sum_total = prefix_sums[-1] if prefix_sums else 0
print(s_min + sum_total)