X = int(input())

# Total sum of all values in the 9x9 grid
total_sum = 45 * 45  # 2025

# Count how many times X appears in the grid
count_X = 0
for i in range(1, 10):
    if X % i == 0:
        j = X // i
        if 1 <= j <= 9:
            count_X += 1

# Sum of values that are not X
result = total_sum - X * count_X
print(result)