s = input()
t = input()
s_idx = 0
result = []
for t_idx, char in enumerate(t):
    if s_idx < len(s) and char == s[s_idx]:
        result.append(t_idx + 1)
        s_idx += 1
print(*result)