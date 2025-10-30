def get_char_at_position(s, k):
    orig_len = len(s)
    
    # After one operation, length doubles
    # Need to find the pattern length and position within pattern
    
    # Create one iteration to understand the pattern
    t = ''.join(c.lower() if c.isupper() else c.upper() for c in s)
    one_iter = s + t
    
    # Pattern length is 2 * original length
    pattern_len = 2 * orig_len
    
    # Adjust k to be 0-based and get position within pattern
    k = k - 1
    pos = k % pattern_len
    
    # If in first half, return from original string
    if pos < orig_len:
        return s[pos]
    # If in second half, return from transformed string
    else:
        pos = pos - orig_len
        return s[pos].lower() if s[pos].isupper() else s[pos].upper()

# Read input
s = input().strip()
q = int(input())
k_values = list(map(int, input().split()))

# Process each query
result = []
for k in k_values:
    result.append(get_char_at_position(s, k))

# Output results
print(' '.join(result))