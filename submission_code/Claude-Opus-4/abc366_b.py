# YOUR CODE HERE
n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Find maximum length
m = max(len(s) for s in strings)

# Create result strings
result = []
for j in range(m):
    # Build T_j character by character
    chars = []
    for i in range(n):
        pos = n - i - 1  # Position in T_j for string S_i (0-indexed)
        if j < len(strings[i]):
            # Character exists in S_i at position j
            chars.append((pos, strings[i][j]))
        else:
            # No character in S_i at position j, use *
            chars.append((pos, '*'))
    
    # Sort by position and build the string
    chars.sort()
    t_j = ''.join(c[1] for c in chars)
    
    # Remove trailing asterisks
    t_j = t_j.rstrip('*')
    
    result.append(t_j)

# Print result
for t in result:
    print(t)