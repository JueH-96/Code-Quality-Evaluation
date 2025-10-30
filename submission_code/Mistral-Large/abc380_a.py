import sys
from collections import Counter

def check_conditions(N):
    # Convert the number to a string to easily count digits
    digits = str(N)
    # Use Counter to count occurrences of each digit
    digit_counts = Counter(digits)

    # Check the conditions
    if digit_counts['1'] == 1 and digit_counts['2'] == 2 and digit_counts['3'] == 3:
        return "Yes"
    else:
        return "No"

# Read input from stdin
N = int(sys.stdin.read().strip())

# Check the conditions and print the result
print(check_conditions(N))