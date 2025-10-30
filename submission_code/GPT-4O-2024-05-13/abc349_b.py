# YOUR CODE HERE
from collections import Counter

def is_good_string(S):
    # Count the frequency of each character in the string
    freq = Counter(S)
    
    # Count the frequency of these frequencies
    freq_of_freq = Counter(freq.values())
    
    # Check the condition for a good string
    for count in freq_of_freq.values():
        if count != 0 and count != 2:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read().strip()

# Print the result
print(is_good_string(input))