def find_first_difference(S, T):
    min_length = min(len(S), len(T))
    
    # Check each character position
    for i in range(min_length):
        if S[i] != T[i]:
            return i + 1  # 1-indexed
    
    # If we got here, the strings match up to min_length
    # Check if lengths differ
    if len(S) != len(T):
        return min_length + 1
    
    # Strings are identical
    return 0

# Read input
S = input().strip()
T = input().strip()

# Output result
print(find_first_difference(S, T))