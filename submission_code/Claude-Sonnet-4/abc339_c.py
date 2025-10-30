# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

prefix_sum = 0
min_prefix_sum = 0
total_sum = 0

for i in range(n):
    prefix_sum += a[i]
    min_prefix_sum = min(min_prefix_sum, prefix_sum)
    total_sum += a[i]

initial_passengers = max(0, -min_prefix_sum)
current_passengers = initial_passengers + total_sum

print(current_passengers)