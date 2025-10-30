def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    t = 0
    enemy_idx = 0
    
    while enemy_idx < n:
        t += 1
        damage = 3 if t % 3 == 0 else 1
        h[enemy_idx] -= damage
        if h[enemy_idx] <= 0:
            enemy_idx += 1
            
    print(t)

solve()