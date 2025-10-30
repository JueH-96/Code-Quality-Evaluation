# YOUR CODE HERE
from collections import Counter

def most_frequent_char(S):
    # Count the frequency of each character in the string
    char_count = Counter(S)
    
    # Find the maximum frequency
    max_freq = max(char_count.values())
    
    # Find the characters with the maximum frequency
    most_frequent_chars = [char for char, count in char_count.items() if count == max_freq]
    
    # Return the earliest character in alphabetical order
    return min(most_frequent_chars)

# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Print the result
print(most_frequent_char(input))