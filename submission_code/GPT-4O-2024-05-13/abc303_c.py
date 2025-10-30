# YOUR CODE HERE
def can_complete_moves(N, M, H, K, S, items):
    health = H
    x, y = 0, 0
    item_set = set(items)
    
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
            return "No"
        
        if (x, y) in item_set and health < K:
            health = K
            item_set.remove((x, y))
    
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = int(data[2])
K = int(data[3])
S = data[4]

items = []
for i in range(M):
    x = int(data[5 + 2 * i])
    y = int(data[6 + 2 * i])
    items.append((x, y))

print(can_complete_moves(N, M, H, K, S, items))