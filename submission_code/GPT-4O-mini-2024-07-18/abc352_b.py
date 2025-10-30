def find_correctly_typed_positions(S, T):
    correct_positions = []
    s_index = 0
    t_index = 0
    
    while s_index < len(S) and t_index < len(T):
        if S[s_index] == T[t_index]:
            correct_positions.append(t_index + 1)  # Store 1-based index
            s_index += 1
        t_index += 1
    
    return correct_positions

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
S = data[0]
T = data[1]

# Get the positions of correctly typed characters
result = find_correctly_typed_positions(S, T)

# Print the result
print(" ".join(map(str, result)))