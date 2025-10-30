# YOUR CODE HERE
X = int(input())

# The 9x9 multiplication table contains 81 integers: i * j for i and j from 1 to 9.
# The sum of all 81 integers in this table is the sum of i * j for all pairs (i, j)
# where i and j range from 1 to 9.
# This sum can be calculated as:
# sum(i * j for i in range(1, 10) for j in range(1, 10))
# which is equal to (sum(i for i in range(1, 10))) * (sum(j for j in range(1, 10))).
# The sum of integers from 1 to 9 is 1 + 2 + ... + 9 = 9 * (9 + 1) // 2 = 45.
# Therefore, the total sum of all 81 entries in the table is 45 * 45 = 2025.
total_sum_all = 2025

# We need to find the sum of the integers that are NOT equal to X among the 81 integers.
# This sum can be obtained by taking the total sum of all 81 integers and subtracting
# the sum of all occurrences of the integer X in the table.
# If the value X appears 'count_x' times in the table, then the sum of the values
# in those specific cells where the value is X is count_x * X.

# Count how many times the value X appears in the 9x9 multiplication table
count_x = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j == X:
            count_x += 1

# Calculate the sum of the integers that are not X
# This is the total sum of all 81 entries minus the sum of the entries that are equal to X.
sum_not_x = total_sum_all - (count_x * X)

# Print the result
print(sum_not_x)