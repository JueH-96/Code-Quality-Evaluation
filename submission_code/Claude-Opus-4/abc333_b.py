# YOUR CODE HERE
def are_adjacent(p1, p2):
    # Define adjacency in a pentagon ABCDE
    adjacency = {
        'A': {'B', 'E'},
        'B': {'A', 'C'},
        'C': {'B', 'D'},
        'D': {'C', 'E'},
        'E': {'D', 'A'}
    }
    return p2 in adjacency[p1]

# Read input
s1s2 = input().strip()
t1t2 = input().strip()

s1, s2 = s1s2[0], s1s2[1]
t1, t2 = t1t2[0], t1t2[1]

# Check if both pairs have the same type of connection
s_adjacent = are_adjacent(s1, s2)
t_adjacent = are_adjacent(t1, t2)

if s_adjacent == t_adjacent:
    print("Yes")
else:
    print("No")