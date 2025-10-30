import sys
from collections import Counter

def is_good_string(s):
    # Count the frequency of each letter
    frequency = Counter(s)

    # Count the frequency of frequencies
    frequency_of_frequencies = Counter(frequency.values())

    # Check the condition for a good string
    for count in frequency_of_frequencies.values():
        if count != 0 and count != 2:
            return "No"

    return "Yes"

# Read input from stdin
s = sys.stdin.read().strip()

# Determine if the string is good and print the result
print(is_good_string(s))