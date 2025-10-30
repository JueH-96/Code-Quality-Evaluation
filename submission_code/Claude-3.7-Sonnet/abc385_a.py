# YOUR CODE HERE
def can_divide_with_equal_sums(A, B, C):
    # Case 1: All three integers are equal (three groups)
    if A == B == C:
        return True
    
    # Case 2: One integer equals the sum of the other two (two groups)
    if A == B + C or B == A + C or C == A + B:
        return True
    
    return False

# Read input
A, B, C = map(int, input().split())

# Print output
if can_divide_with_equal_sums(A, B, C):
    print("Yes")
else:
    print("No")