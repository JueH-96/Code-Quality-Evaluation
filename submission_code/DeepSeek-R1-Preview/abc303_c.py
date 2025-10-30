n, m, h, k = map(int, input().split())
s = input().strip()
items = set()
for _ in range(m):
    x, y = map(int, input().split())
    items.add((x, y))

current_x, current_y = 0, 0
current_health = h

for c in s:
    # Move
    if c == 'R':
        current_x += 1
    elif c == 'L':
        current_x -= 1
    elif c == 'U':
        current_y += 1
    elif c == 'D':
        current_y -= 1
    
    # Consume health
    current_health -= 1
    
    # Check health
    if current_health < 0:
        print("No")
        exit()
    
    # Check for item
    pos = (current_x, current_y)
    if pos in items and current_health < k:
        items.remove(pos)
        current_health = k

print("Yes")