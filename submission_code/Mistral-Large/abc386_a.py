import sys
from collections import Counter

def can_form_full_house(A, B, C, D):
    # Count the occurrences of each card
    counts = Counter([A, B, C, D])

    # Check if we can form a Full House by adding one card
    for value, count in counts.items():
        if count == 3:
            # We need two of another value
            for other_value, other_count in counts.items():
                if other_value != value and other_count == 1:
                    return "Yes"
        elif count == 2:
            # We need three of another value
            for other_value, other_count in counts.items():
                if other_value != value and other_count == 2:
                    return "Yes"

    return "No"

# Read input from stdin
input_line = sys.stdin.readline().strip()
A, B, C, D = map(int, input_line.split())

# Determine if a Full House can be formed and print the result
result = can_form_full_house(A, B, C, D)
print(result)