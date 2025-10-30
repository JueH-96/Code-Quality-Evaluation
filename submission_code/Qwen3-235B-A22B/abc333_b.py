# Read input lines
s = input().strip()
t = input().strip()

# Mapping of each point to its position (0 for A, 1 for B, etc.)
position = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def min_step(a, b):
    """Calculate the minimal step distance between two points."""
    pos_a = position[a]
    pos_b = position[b]
    diff = abs(pos_a - pos_b)
    return min(diff, 5 - diff)

# Calculate for both segments
s1, s2 = s[0], s[1]
t1, t2 = t[0], t[1]

s_length = min_step(s1, s2)
t_length = min_step(t1, t2)

# Output result
print("Yes" if s_length == t_length else "No")