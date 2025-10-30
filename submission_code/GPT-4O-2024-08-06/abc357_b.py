# YOUR CODE HERE
def process_string(S):
    # Count the number of uppercase and lowercase letters
    uppercase_count = sum(1 for c in S if c.isupper())
    lowercase_count = len(S) - uppercase_count  # Since the rest are lowercase

    # Determine which conversion to apply
    if uppercase_count > lowercase_count:
        # Convert all to uppercase
        return S.upper()
    else:
        # Convert all to lowercase
        return S.lower()

# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Process the input string
result = process_string(input)

# Print the result
print(result)