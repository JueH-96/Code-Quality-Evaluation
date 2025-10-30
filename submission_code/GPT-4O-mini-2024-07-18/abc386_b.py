def min_button_presses(S):
    # Initialize the count of button presses
    presses = 0
    # Initialize the current position in the string
    i = 0
    n = len(S)
    
    while i < n:
        # Check if we can take "00" (two zeros) from the current position
        if i + 1 < n and S[i] == '0' and S[i + 1] == '0':
            presses += 1  # Press the button for "00"
            i += 2  # Move past these two zeros
        else:
            presses += 1  # Press the button for the current digit
            i += 1  # Move to the next character
            
    return presses

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Calculate the minimum button presses
result = min_button_presses(S)

# Print the result
print(result)