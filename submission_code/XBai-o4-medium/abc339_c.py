n = int(input())
a = list(map(int, input().split()))

current_sum = 0
min_prefix = 0

for num in a:
    current_sum += num
    if current_sum < min_prefix:
        min_prefix = current_sum

required_initial = max(0, -min_prefix)
total_sum = current_sum

print(required_initial + total_sum)