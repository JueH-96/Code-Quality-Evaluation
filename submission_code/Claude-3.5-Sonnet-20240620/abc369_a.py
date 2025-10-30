# YOUR CODE HERE
def count_arithmetic_sequence(A, B):
    count = 0
    
    # Check all possible x values
    for x in range(-100, 201):  # Expanded range to cover all possible x
        # Check all permutations of A, B, x
        if (
            (B - A == x - B) or (A - B == x - A) or  # A, B, x
            (B - x == A - B) or (x - B == A - x) or  # B, x, A
            (A - x == B - A) or (x - A == B - x)     # x, A, B
        ):
            count += 1
    
    return count

# Read input
A, B = map(int, input().split())

# Calculate and print result
result = count_arithmetic_sequence(A, B)
print(result)