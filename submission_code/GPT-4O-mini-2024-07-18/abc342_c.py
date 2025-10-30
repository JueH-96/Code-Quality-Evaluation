def replace_characters(N, S, Q, operations):
    # Create a mapping for character replacements
    replacement_map = {}
    
    # Process each operation
    for c, d in operations:
        if c in replacement_map:
            # If c is already mapped to another character, update the mapping
            current_mapping = replacement_map[c]
            replacement_map[d] = current_mapping
            replacement_map.pop(c)
        else:
            replacement_map[c] = d
    
    # Create a new string based on the replacement map
    result = []
    for char in S:
        if char in replacement_map:
            result.append(replacement_map[char])
        else:
            result.append(char)
    
    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
S = data[1]
Q = int(data[2])
operations = [line.split() for line in data[3:3 + Q]]

# Get the result after all operations
result = replace_characters(N, S, Q, operations)

# Print the result
print(result)