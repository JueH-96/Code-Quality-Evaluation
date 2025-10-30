n, m, h, k = map(int, input().split())
s = input().strip()

items = set()
for _ in range(m):
    x, y = map(int, input().split())
    items.add((x, y))

current_x, current_y = 0, 0
health = h

for c in s:
    # Move according to the current character
    if c == 'R':
        current_x += 1
    elif c == 'L':
        current_x -= 1
    elif c == 'U':
        current_y += 1
    elif c == 'D':
        current_y -= 1
    
    # Decrease health
    health -= 1
    if health < 0:
        print("No")
        exit()
    
    # Check for item and use it if applicable
    pos = (current_x, current_y)
    if pos in items:
        if health < k:
            health = k
            items.remove(pos)

print("Yes")