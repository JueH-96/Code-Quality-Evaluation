# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Calculate total sum
total_sum = sum(a)

# Find target values
x = total_sum // n
count_x_plus_1 = total_sum % n
count_x = n - count_x_plus_1

# Count how many operations needed
decrease_needed = 0
increase_needed = 0

for val in a:
    if val > x + 1:
        decrease_needed += val - (x + 1)
    elif val < x:
        increase_needed += x - val

# The answer is the maximum of decrease and increase needed
# (they should be equal in a valid redistribution)
print(max(decrease_needed, increase_needed))