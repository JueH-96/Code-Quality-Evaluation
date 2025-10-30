# YOUR CODE HERE
import sys

# Read the input line as a string.
# .strip() is used to remove any leading/trailing whitespace, including the newline character.
x_str = sys.stdin.readline().strip()

# Step 1: Remove trailing zeros from the string.
# Example: "12.340" -> "12.34"
# Example: "99.900" -> "99.9"
# Example: "0.000" -> "0."
cleaned_str = x_str.rstrip('0')

# Step 2: If the resulting string ends with a decimal point, remove it.
# This handles cases where the number effectively became an integer.
# Example: "0." -> "0"
# Example: "5." (if input was "5.000") -> "5"
if cleaned_str.endswith('.'):
    cleaned_str = cleaned_str.rstrip('.')

# Print the final formatted string.
print(cleaned_str)