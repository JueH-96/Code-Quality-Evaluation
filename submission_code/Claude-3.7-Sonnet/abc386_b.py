# YOUR CODE HERE
def min_button_presses(s):
    total_presses = 0
    consecutive_zeros = 0
    
    for char in s:
        if char == '0':
            consecutive_zeros += 1
        else:
            # Handle any consecutive zeros we've seen before this non-zero digit
            total_presses += (consecutive_zeros + 1) // 2
            consecutive_zeros = 0
            # Add one press for the current non-zero digit
            total_presses += 1
    
    # Handle any trailing zeros at the end of the string
    total_presses += (consecutive_zeros + 1) // 2
    
    return total_presses

# Read input
S = input().strip()

# Print output
print(min_button_presses(S))