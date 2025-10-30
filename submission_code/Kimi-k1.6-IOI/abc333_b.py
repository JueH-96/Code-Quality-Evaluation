# Read the input pairs
s = input().strip()
t = input().strip()

# Mapping each character to an index (0-4)
char_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def calculate_min_steps(pair):
    # Convert characters to their respective indices
    idx1 = char_to_index[pair[0]]
    idx2 = char_to_index[pair[1]]
    # Calculate the absolute difference and find the minimal steps
    difference = abs(idx1 - idx2)
    return min(difference, 5 - difference)

# Calculate minimal steps for both pairs
s_steps = calculate_min_steps(s)
t_steps = calculate_min_steps(t)

# Compare and print the result
print("Yes" if s_steps == t_steps else "No")