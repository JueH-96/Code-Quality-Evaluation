import sys

# Read input from stdin
A, B, C = map(int, input().split())

# Calculate the total sum
total_sum = A + B + C

# Check if the total sum is divisible by 2
if total_sum % 2 != 0:
    print("No")
    sys.exit()

# Check if it's possible to divide the numbers into two groups with equal sums
target_sum = total_sum // 2
if (A <= target_sum and B <= target_sum and C <= target_sum) or \
   (A <= target_sum and (B + C) == target_sum) or \
   (B <= target_sum and (A + C) == target_sum) or \
   (C <= target_sum and (A + B) == target_sum):
    print("Yes")
else:
    print("No")