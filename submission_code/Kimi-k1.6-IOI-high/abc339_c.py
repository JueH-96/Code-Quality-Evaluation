n = int(input())
a = list(map(int, input().split()))

current_sum = 0
min_sum = 0

for num in a:
    current_sum += num
    if current_sum < min_sum:
        min_sum = current_sum

x = max(0, -min_sum)
print(x + current_sum)