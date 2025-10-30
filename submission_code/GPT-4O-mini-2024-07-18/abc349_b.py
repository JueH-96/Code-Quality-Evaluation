def is_good_string(s):
    from collections import Counter
    
    # Count the frequency of each character in the string
    frequency = Counter(s)
    
    # Count how many letters have each frequency
    freq_count = Counter(frequency.values())
    
    # Check the conditions for a good string
    for count in freq_count:
        if freq_count[count] != 0 and freq_count[count] != 2:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
s = input().strip()

# Output the result
print(is_good_string(s))