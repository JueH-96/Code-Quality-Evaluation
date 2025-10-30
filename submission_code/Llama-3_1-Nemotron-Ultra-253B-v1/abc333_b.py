s = input().strip()
t = input().strip()

def get_step(c1, c2):
    a = ord(c1) - ord('A')
    b = ord(c2) - ord('A')
    diff = abs(a - b)
    return min(diff, 5 - diff)

s_step = get_step(s[0], s[1])
t_step = get_step(t[0], t[1])

print("Yes" if s_step == t_step else "No")