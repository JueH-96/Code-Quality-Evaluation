def can_divide_into_equal_groups(A, B, C):
    # Calculate the total sum of A, B, C
    total_sum = A + B + C
    
    # Check if the total sum can be divided into two or more equal groups
    # We need to check if we can form groups with sums that are divisors of total_sum
    # The possible sums for groups can be 1 to total_sum // 2 (since we need at least two groups)
    
    # Check for all possible group sums from 1 to total_sum // 2
    for group_sum in range(1, total_sum // 2 + 1):
        if total_sum % group_sum == 0:
            # Check if we can form groups with this group_sum
            # The number of groups we can form
            num_groups = total_sum // group_sum
            
            # We need at least 2 groups
            if num_groups >= 2:
                # Now we need to check if we can form these groups
                # We can use a simple check to see if we can form the required sum
                # by checking combinations of A, B, C
                if (A == group_sum or B == group_sum or C == group_sum or
                    (A + B == group_sum) or (A + C == group_sum) or (B + C == group_sum)):
                    return "Yes"
    
    return "No"

# Read input
import sys
input = sys.stdin.read
A, B, C = map(int, input().strip().split())

# Get the result
result = can_divide_into_equal_groups(A, B, C)

# Print the result
print(result)