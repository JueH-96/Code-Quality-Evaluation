n, m = map(int, input().split())
s = list(input().strip())
c_list = list(map(int, input().split()))

color_positions = [[] for _ in range(m + 1)]  # 1-based indexing for colors

for idx in range(n):
    color = c_list[idx]
    color_positions[color].append(idx)

for color in range(1, m + 1):
    pos_list = color_positions[color]
    k = len(pos_list)
    if k <= 1:
        continue
    current_chars = [s[p] for p in pos_list]
    new_chars = [current_chars[-1]] + current_chars[:-1]
    for i in range(k):
        s[pos_list[i]] = new_chars[i]

print(''.join(s))