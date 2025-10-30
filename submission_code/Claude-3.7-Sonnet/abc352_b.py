def find_correctly_typed_positions(S, T):
    positions = []
    s_index = 0
    
    for t_index, char in enumerate(T):
        if s_index < len(S) and char == S[s_index]:
            positions.append(t_index + 1)  # Adding 1 because positions are 1-indexed
            s_index += 1
    
    return positions

# Read the input
S = input().strip()
T = input().strip()

# Find positions of correctly typed characters
positions = find_correctly_typed_positions(S, T)

# Output the positions
print(" ".join(map(str, positions)))