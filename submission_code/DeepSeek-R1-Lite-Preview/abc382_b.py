# Read input values
N_D = input().strip().split()
N = int(N_D[0])
D = int(N_D[1])
S = input().strip()

# Convert S to a list for easy modification
boxes = list(S)

# Find positions of '@' (1-based indexing)
at_positions = [i+1 for i in range(N) if boxes[i] == '@']

# Sort positions in decreasing order
at_positions_sorted = sorted(at_positions, reverse=True)

# Select the first D positions to eat
positions_to_eat = at_positions_sorted[:D]

# Set these positions to '.'
for pos in positions_to_eat:
    boxes[pos-1] = '.'  # Convert to 0-based index

# Join the list back into a string and print
final_string = ''.join(boxes)
print(final_string)