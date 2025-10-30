# YOUR CODE HERE
h, w, m = map(int, input().split())
rows = [0] * h
cols = [0] * w
colors = set()

for _ in range(m):
    t, a, x = map(int, input().split())
    a -= 1
    if t == 1:
        if x not in colors:
            colors.add(x)
        for i in range(w):
            if rows[a] != x:
                rows[a] = x
                colors.add(x)
    else:
        if x not in colors:
            colors.add(x)
        for i in range(h):
            if cols[a] != x:
                cols[a] = x
                colors.add(x)

color_count = {}
for i in range(h):
    if rows[i] != 0:
        color_count[rows[i]] = color_count.get(rows[i], 0) + 1
for i in range(w):
    if cols[i] != 0:
        color_count[cols[i]] = color_count.get(cols[i], 0) + 1

print(len(colors))
for color in sorted(colors):
    print(color, color_count[color])