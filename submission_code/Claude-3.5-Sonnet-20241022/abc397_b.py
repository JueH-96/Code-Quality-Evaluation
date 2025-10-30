# Read input
S = input().strip()

def min_insertions(s):
    n = len(s)
    min_inserts = 0
    
    # Create the target pattern (alternating i and o)
    # We'll extend this pattern to be longer than input string
    pattern = 'io' * ((n + 2) // 2)
    
    # Current position in the input string
    curr_pos = 0
    # Position in the pattern we're trying to match
    pattern_pos = 0
    
    while curr_pos < n:
        # If current character doesn't match pattern
        if s[curr_pos] != pattern[pattern_pos]:
            # We need to insert a character here
            min_inserts += 1
            # Move only in pattern (simulating insertion)
            pattern_pos += 1
        else:
            # Characters match, move both pointers
            curr_pos += 1
            pattern_pos += 1
    
    # If we haven't reached an even length or pattern isn't complete
    if (curr_pos + min_inserts) % 2 != 0:
        min_inserts += 1
        
    return min_inserts

# Print result
print(min_insertions(S))