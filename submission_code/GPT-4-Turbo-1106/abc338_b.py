from collections import Counter

def find_most_frequent_char(s):
    # Count the frequency of each character in the string
    char_count = Counter(s)
    # Find the maximum frequency
    max_freq = max(char_count.values())
    # Filter characters with the maximum frequency and sort them alphabetically
    most_frequent_chars = sorted([char for char, freq in char_count.items() if freq == max_freq])
    # Return the first character in alphabetical order
    return most_frequent_chars[0]

# Read input from stdin
S = input().strip()

# Find and print the result
print(find_most_frequent_char(S))