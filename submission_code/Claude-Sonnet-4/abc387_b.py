# YOUR CODE HERE
X = int(input())

# Calculate total sum of all values in 9x9 multiplication table
total_sum = 45 * 45  # (1+2+...+9) * (1+2+...+9) = 2025

# Count how many times X appears in the grid
count_x = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j == X:
            count_x += 1

# Sum of values that are not X = total_sum - (X * count_x)
result = total_sum - X * count_x

print(result)