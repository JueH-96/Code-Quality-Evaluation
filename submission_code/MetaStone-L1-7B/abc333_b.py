positions = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def get_min_steps(pair):
    p = positions[pair[0]]
    q = positions[pair[1]]
    difference = abs(p - q)
    return min(difference, 5 - difference)

s = input().strip()
t = input().strip()

s_min = get_min_steps(s)
t_min = get_min_steps(t)

print("Yes" if s_min == t_min else "No")