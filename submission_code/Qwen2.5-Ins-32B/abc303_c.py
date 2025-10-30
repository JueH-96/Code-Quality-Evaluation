import sys

def can_complete_moves(N, M, H, K, S, items):
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
            return "No"
        if (x, y) in items and health < K:
            health = K
    return "Yes"

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    H = int(data[2])
    K = int(data[3])
    S = data[4]
    
    items = set()
    for i in range(M):
        x = int(data[5 + 2 * i])
        y = int(data[6 + 2 * i])
        items.add((x, y))
    
    print(can_complete_moves(N, M, H, K, S, items))