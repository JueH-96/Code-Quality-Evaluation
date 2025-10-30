n = int(input())
s_list = [input().strip() for _ in range(n)]
m = max(len(s) for s in s_list) if s_list else 0

for j in range(1, m + 1):
    required_positions = []
    for i in range(n):
        s = s_list[i]
        if j <= len(s):
            pos = n - i
            required_positions.append(pos)
    if not required_positions:
        continue
    max_pos = max(required_positions)
    t = ['*'] * max_pos
    for i in range(n):
        s = s_list[i]
        if j <= len(s):
            pos = n - i
            t[pos - 1] = s[j - 1]
    t_str = ''.join(t).rstrip('*')
    print(t_str)