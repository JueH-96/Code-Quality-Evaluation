s = input().strip()
t = input().strip()

max_len = max(len(s), len(t))

for i in range(1, max_len + 1):
    s_char = s[i-1] if i <= len(s) else None
    t_char = t[i-1] if i <= len(t) else None
    if s_char != t_char:
        print(i)
        break
else:
    print(0)