n = int(input())
a = list(map(int, input().split()))
sum_a = sum(a)
current_sum = 0
min_prefix = 0
for num in a:
    current_sum += num
    if current_sum < min_prefix:
        min_prefix = current_sum
initial = max(0, -min_prefix)
print(initial + sum_a)