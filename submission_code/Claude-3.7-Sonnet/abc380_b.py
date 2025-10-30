def reconstruct_sequence(s):
    # Split the string by '|'
    parts = s.split('|')
    
    # Skip the first and last elements (which are empty)
    parts = parts[1:-1]
    
    # Count the number of '-' characters in each part
    result = [len(part) for part in parts]
    
    return result

# Read input from stdin
S = input().strip()

# Reconstruct the sequence
A = reconstruct_sequence(S)

# Print the result with elements separated by spaces
print(' '.join(map(str, A)))