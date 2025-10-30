import math

def solve(K, sx, sy, tx, ty):
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    base = math.ceil(dx / K) + math.ceil(dy / K)
    
    # Adjustment: if we need to move in both x and y directions,
    # and one of the distances is less than K, we might need an extra move
    if dx > 0 and dy > 0 and (dx < K or dy < K):
        base += 1
    
    return base

T = int(input())
for _ in range(T):
    K, sx, sy, tx, ty = map(int, input().split())
    print(solve(K, sx, sy, tx, ty))