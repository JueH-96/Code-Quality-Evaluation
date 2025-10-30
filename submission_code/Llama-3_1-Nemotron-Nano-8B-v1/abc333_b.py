# YOUR CODE HERE
positions = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

s = input().strip()
t = input().strip()

def compute_step(pair):
    a = positions[pair[0]]
    b = positions[pair[1]]
    diff = abs(a - b)
    return min(diff, 5 - diff)

s_step = compute_step(s)
t_step = compute_step(t)

print("Yes" if (s_step == 1 and t_step == 1) or (s_step == 2 and t_step == 2) else "No")