# YOUR CODE HERE
h, w, n = map(int, input().split())
t = input()
s = [input() for _ in range(h)]

x, y = 0, 0
for d in t:
    if d == 'L' and y > 0 and s[x][y-1] != '#':
        y -= 1
    elif d == 'R' and y < w-1 and s[x][y+1] != '#':
        y += 1
    elif d == 'U' and x > 0 and s[x-1][y] != '#':
        x -= 1
    elif d == 'D' and x < h-1 and s[x+1][y] != '#':
        x += 1

ans = 0
for i in range(h):
    for j in range(w):
        if s[i][j] == '.' and (i == x or j == y) and (i == 0 or s[i-1][j] == '#') and (i == h-1 or s[i+1][j] == '#') and (j == 0 or s[i][j-1] == '#') and (j == w-1 or s[i][j+1] == '#'):
            ans += 1

print(ans)