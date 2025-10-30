# Read input from stdin
X = int(input())

# Total sum of all elements in 9x9 multiplication table
total_sum = 2025

# Count the number of times X appears in the grid
count = 0
for i in range(1, 10):  # i from 1 to 9
    if X % i == 0:
        j = X // i
        if 1 <= j <= 9:
            count += 1

# Calculate the sum excluding all occurrences of X
result = total_sum - X * count

# Output the result
print(result)