import collections

# Read the input string S
S = input()

# Count the frequency of each character in S
counts = collections.Counter(S)

# Find the maximum frequency.
# Constraints: 1 <= |S| <= 1000, so S is not empty, counts is not empty,
# and counts.values() is not empty. Thus, max() is safe.
max_freq = max(counts.values())

# Initialize a variable to store the result character.
# It will be updated when the correct character is found.
result_char = ''

# Iterate through all lowercase English letters from 'a' to 'z'
# to find the alphabetically earliest character with the maximum frequency.
for i in range(26):  # This loop will iterate 26 times, for 'a' through 'z'
    # Get the current character to check (e.g., 'a', then 'b', etc.)
    char_to_check = chr(ord('a') + i)
    
    # Check if this character's frequency (from the counter) is equal to max_freq.
    # counts.get(char, 0) returns the frequency of char, or 0 if char is not in S.
    if counts.get(char_to_check, 0) == max_freq:
        result_char = char_to_check
        # Since we are iterating in alphabetical order ('a', 'b', 'c', ...),
        # the first character we find that has the max_freq is the one we want.
        # So, we can break the loop.
        break

# Print the result character
print(result_char)