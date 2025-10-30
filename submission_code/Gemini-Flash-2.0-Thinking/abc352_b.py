s = input()
t = input()
n_s = len(s)
n_t = len(t)

correct_positions = []
t_ptr = 0

for s_char in s:
    found = False
    for i in range(t_ptr, n_t):
        if t[i] == s_char:
            correct_positions.append(i + 1)
            t_ptr = i + 1
            found = True
            break
    if not found:
        raise ValueError("Could not find matching character in T")

print(*correct_positions)