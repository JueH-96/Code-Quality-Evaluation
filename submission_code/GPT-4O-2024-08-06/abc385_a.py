# YOUR CODE HERE
def can_divide_into_equal_groups(A, B, C):
    # Calculate the total sum of A, B, C
    total_sum = A + B + C
    
    # Check if the total sum can be divided into two equal parts
    # This is only possible if the total sum is even
    if total_sum % 2 != 0:
        return "No"
    
    # Check all possible groupings
    # We need to check if there exists a subset with sum equal to total_sum / 2
    half_sum = total_sum // 2
    
    # Check if any single number is equal to half_sum
    if A == half_sum or B == half_sum or C == half_sum:
        return "Yes"
    
    # Check if any two numbers sum up to half_sum
    if A + B == half_sum or A + C == half_sum or B + C == half_sum:
        return "Yes"
    
    # If none of the above conditions are met, it's not possible
    return "No"

# Read input
import sys
input = sys.stdin.read
data = input().strip()

# Parse input
A, B, C = map(int, data.split())

# Determine if it's possible to divide into equal groups
result = can_divide_into_equal_groups(A, B, C)

# Print the result
print(result)