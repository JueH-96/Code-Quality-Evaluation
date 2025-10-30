# Read the input pairs
s_pair = input().strip()
t_pair = input().strip()

# Mapping of each point to its index
point_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def calculate_min_steps(pair):
    a, b = pair[0], pair[1]
    x = point_index[a]
    y = point_index[b]
    diff = abs(x - y)
    return min(diff, 5 - diff)

# Calculate minimal steps for both pairs
s_steps = calculate_min_steps(s_pair)
t_steps = calculate_min_steps(t_pair)

# Compare and output the result
print("Yes" if s_steps == t_steps else "No")