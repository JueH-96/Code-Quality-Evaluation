# Read input
s = input().strip()
t = input().strip()

# Mapping of points to numerical values for easier calculation
point_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def is_adjacent(c1, c2):
    x = point_map[c1]
    y = point_map[c2]
    return (x + 1) % 5 == y or (x - 1) % 5 == y

# Check if both pairs are adjacent or both are non-adjacent
s_adjacent = is_adjacent(s[0], s[1])
t_adjacent = is_adjacent(t[0], t[1])

print("Yes" if s_adjacent == t_adjacent else "No")