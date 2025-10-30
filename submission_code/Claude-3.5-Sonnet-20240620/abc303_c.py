# YOUR CODE HERE
N, M, H, K = map(int, input().split())
S = input().strip()
items = set(tuple(map(int, input().split())) for _ in range(M))

x, y = 0, 0
health = H

for move in S:
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    
    health -= 1
    
    if health < 0:
        print("No")
        exit()
    
    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))

print("Yes")