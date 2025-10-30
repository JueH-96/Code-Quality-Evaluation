# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

current_sum = 0
min_val = 0
for x in a:
    current_sum += x
    min_val = max(min_val, -current_sum)

print(min_val)