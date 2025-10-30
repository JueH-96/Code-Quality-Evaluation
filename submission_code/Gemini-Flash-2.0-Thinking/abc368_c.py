def solve():
    n = int(input())
    h = list(map(int, input().split()))

    t = 0
    while any(health > 0 for health in h):
        t += 1
        frontmost_enemy_index = -1
        for i in range(n):
            if h[i] > 0:
                frontmost_enemy_index = i
                break

        if frontmost_enemy_index != -1:
            damage = 3 if t % 3 == 0 else 1
            h[frontmost_enemy_index] -= damage

    print(t)

solve()