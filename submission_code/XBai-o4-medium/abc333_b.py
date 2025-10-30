# Read input pairs
mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

s_pair = input().strip()
t_pair = input().strip()

def get_min_step(pair):
    c1, c2 = pair[0], pair[1]
    i1 = mapping[c1]
    i2 = mapping[c2]
    diff = abs(i1 - i2)
    return min(diff, 5 - diff)

s_step = get_min_step(s_pair)
t_step = get_min_step(t_pair)

print("Yes" if s_step == t_step else "No")