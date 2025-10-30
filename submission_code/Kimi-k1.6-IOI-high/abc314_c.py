n, m = map(int, input().split())
s = list(input().strip())
c_list = list(map(int, input().split()))

# Create groups of indices for each color
groups = [[] for _ in range(m + 1)]  # colors are 1-based
for idx in range(n):
    color = c_list[idx]
    groups[color].append(idx)

# Process each color from 1 to m
for color in range(1, m + 1):
    group = groups[color]
    if not group:
        continue
    # Extract current characters in the group
    chars = [s[pos] for pos in group]
    # Perform right circular shift
    new_chars = [chars[-1]] + chars[:-1]
    # Update the characters in the original array
    for i in range(len(group)):
        s[group[i]] = new_chars[i]

print(''.join(s))