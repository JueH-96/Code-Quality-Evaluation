def get_step(a, b):
    a_num = ord(a) - ord('A')
    b_num = ord(b) - ord('A')
    diff = abs(a_num - b_num)
    return min(diff, 5 - diff)

# Read input
s_line = input().strip()
t_line = input().strip()

s1, s2 = s_line[0], s_line[1]
t1, t2 = t_line[0], t_line[1]

s_step = get_step(s1, s2)
t_step = get_step(t1, t2)

print("Yes" if s_step == t_step else "No")