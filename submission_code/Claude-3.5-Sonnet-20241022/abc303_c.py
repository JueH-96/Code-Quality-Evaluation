N, M, H, K = map(int, input().split())
S = input()
items = set()
for _ in range(M):
    x, y = map(int, input().split())
    items.add((x,y))

x = 0
y = 0
health = H

for c in S:
    if health <= 0:
        print("No")
        exit()
        
    if c == 'R':
        x += 1
    elif c == 'L':
        x -= 1
    elif c == 'U':
        y += 1
    else:  # D
        y -= 1
        
    health -= 1
    
    if health < 0:
        print("No")
        exit()
        
    if health < K and (x,y) in items:
        health = K
        items.remove((x,y))

print("Yes")