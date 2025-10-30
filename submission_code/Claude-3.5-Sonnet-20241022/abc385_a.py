def can_divide_equal_sums(A, B, C):
    # Try all possible combinations of groupings
    numbers = [A, B, C]
    n = len(numbers)
    
    # Check two groups
    # Try each possible combination of numbers in first group
    for i in range(1 << n):
        group1_sum = 0
        group2_sum = 0
        # If we have at least one number in each group
        group1_count = 0
        group2_count = 0
        
        for j in range(n):
            if i & (1 << j):
                group1_sum += numbers[j]
                group1_count += 1
            else:
                group2_sum += numbers[j]
                group2_count += 1
        
        # Check if both groups are non-empty and have equal sums
        if group1_count > 0 and group2_count > 0 and group1_sum == group2_sum:
            return True
    
    # Check three groups (only possible if all numbers are equal)
    if A == B and B == C:
        return True
    
    return False

# Read input
A, B, C = map(int, input().split())

# Print result
if can_divide_equal_sums(A, B, C):
    print("Yes")
else:
    print("No")