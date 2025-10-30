# YOUR CODE HERE
def compare_strings(s, t):
    min_length = min(len(s), len(t))
    
    # Compare characters
    for i in range(min_length):
        if s[i] != t[i]:
            return i + 1  # Adding 1 because positions are 1-indexed
    
    # If we've reached here, all characters up to min_length are the same
    if len(s) != len(t):
        return min_length + 1
    
    # If we've reached here, the strings are identical
    return 0

# Read input
s = input().strip()
t = input().strip()

# Compare strings and print result
result = compare_strings(s, t)
print(result)