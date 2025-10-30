from collections import Counter

def find_unique_character_position(S):
    # Count the frequency of each character
    freq = Counter(S)
    
    # Iterate through the string to find the character with frequency 1
    for index, char in enumerate(S):
        if freq[char] == 1:
            # Return 1-based position
            return index + 1

# Read input from standard input
import sys
S = sys.stdin.read().strip()

# Find and print the position of the unique character
print(find_unique_character_position(S))