from collections import Counter

def most_frequent_character(S):
    # Count the frequency of each character in the string
    frequency = Counter(S)
    
    # Find the maximum frequency
    max_freq = max(frequency.values())
    
    # Filter characters that have the maximum frequency
    most_frequent_chars = [char for char, count in frequency.items() if count == max_freq]
    
    # Return the earliest character in alphabetical order
    return min(most_frequent_chars)

# Read input
S = input().strip()

# Get the result and print it
result = most_frequent_character(S)
print(result)