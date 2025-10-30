from collections import Counter

def is_good_string(S):
    # Count the frequency of each character
    freq = Counter(S)
    
    # Count how many characters have the same frequency
    count_freq = Counter(freq.values())
    
    # Get the maximum frequency to iterate up to
    max_freq = max(freq.values())
    
    # Check for each frequency i from 1 to max_freq
    for i in range(1, max_freq + 1):
        if count_freq[i] not in [0, 2]:
            return "No"
    
    # If all frequencies satisfy the condition, return "Yes"
    return "Yes"

# Read input
S = input().strip()

# Determine if it's a good string
result = is_good_string(S)

# Print the result
print(result)