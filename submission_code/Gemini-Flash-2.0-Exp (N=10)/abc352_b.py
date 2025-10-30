s = input()
t = input()

result = []
s_idx = 0
t_idx = 0

while s_idx < len(s) and t_idx < len(t):
    if s[s_idx] == t[t_idx]:
        result.append(t_idx + 1)
        s_idx += 1
        t_idx += 1
    else:
        t_idx += 1
        
print(*result)