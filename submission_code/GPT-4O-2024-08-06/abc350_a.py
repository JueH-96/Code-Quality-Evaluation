# YOUR CODE HERE
def is_valid_contest(S):
    # Extract the numeric part of the string
    contest_number = int(S[3:])
    
    # Check if the contest number is within the valid range
    if 1 <= contest_number <= 349 and contest_number != 316:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Output the result
print(is_valid_contest(S))