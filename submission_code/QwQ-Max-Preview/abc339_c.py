n = int(input())
a = list(map(int, input().split()))

sum_so_far = 0
min_prefix = sum_so_far  # Initialize with S_0 = 0

for num in a:
    sum_so_far += num
    if sum_so_far < min_prefix:
        min_prefix = sum_so_far

x = max(-min_prefix, 0)
current_passengers = x + sum_so_far

print(current_passengers)