n = int(input())
a = list(map(int, input().split()))

current_sum = 0
min_initial = 0
for x in a:
    current_sum += x
    min_initial = max(min_initial, -current_sum)

print(current_sum + min_initial)