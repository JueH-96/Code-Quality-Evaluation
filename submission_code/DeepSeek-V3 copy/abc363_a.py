# Read the input value for R
R = int(input())

# Determine the current number of ^
if 1 <= R <= 99:
    current_carets = 1
elif 100 <= R <= 199:
    current_carets = 2
elif 200 <= R <= 299:
    current_carets = 3
elif 300 <= R <= 399:
    current_carets = 4

# Determine the next threshold
if current_carets == 1:
    next_threshold = 100
elif current_carets == 2:
    next_threshold = 200
elif current_carets == 3:
    next_threshold = 300

# Calculate the minimum increase required
min_increase = next_threshold - R

# Print the result
print(min_increase)