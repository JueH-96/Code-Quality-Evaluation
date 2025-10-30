# Read the input
s = input().strip()
t = input().strip()

# Define the adjacency for each vertex in the pentagon
adjacent = {
    'A': {'B', 'E'},
    'B': {'A', 'C'},
    'C': {'B', 'D'},
    'D': {'C', 'E'},
    'E': {'D', 'A'}
}

def is_adjacent(a, b):
    return b in adjacent[a]

# Check if the segments are adjacent or not
s_adj = is_adjacent(s[0], s[1])
t_adj = is_adjacent(t[0], t[1])

# Determine if both are edges or both are diagonals
if (s_adj and t_adj) or (not s_adj and not t_adj):
    print("Yes")
else:
    print("No")