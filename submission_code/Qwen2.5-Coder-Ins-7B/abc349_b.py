# YOUR CODE HERE
import sys

def is_good_string(s):
    from collections import Counter
    count = Counter(s)
    freq_count = Counter(count.values())
    
    if len(freq_count) > 2:
        return "No"
    
    if 1 in freq_count and freq_count[1] != 2:
        return "No"
    
    if 2 in freq_count and freq_count[2] != 2:
        return "No"
    
    return "Yes"

# Read input from stdin
s = sys.stdin.read().strip()

# Print the result
print(is_good_string(s))