# YOUR CODE HERE
import sys

# Read input string
S = sys.stdin.readline().strip()

# Count character frequencies
counts = {}
for char in S:
    counts[char] = counts.get(char, 0) + 1

# Find the maximum frequency
# Constraints guarantee |S| >= 1, so counts will not be empty and max_freq will be >= 1
max_freq = max(counts.values())

# Find the character with the maximum frequency that comes earliest alphabetically
result_char = None # Initialize, will be set inside the loop

# Iterate through characters 'a' through 'z' in alphabetical order
for char_code in range(ord('a'), ord('z') + 1):
    char = chr(char_code)

    # Check if this character's count matches the max_freq
    # Use .get(char, 0) to safely check characters not in the original string
    # If a character is not in S, its count is effectively 0, which won't equal max_freq (since max_freq >= 1)
    if counts.get(char, 0) == max_freq:
        result_char = char
        break # Found the alphabetically earliest character with max frequency

# Print the result
print(result_char)