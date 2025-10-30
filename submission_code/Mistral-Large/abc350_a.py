import sys

def is_valid_abbreviation(S):
    # Extract the numeric part of the string
    num_part = int(S[3:])

    # Check if the number is within the valid range and not equal to 316
    if 1 <= num_part <= 349 and num_part != 316:
        return "Yes"
    else:
        return "No"

# Read input from stdin
S = sys.stdin.read().strip()

# Determine if S is a valid abbreviation and print the result
result = is_valid_abbreviation(S)
print(result)