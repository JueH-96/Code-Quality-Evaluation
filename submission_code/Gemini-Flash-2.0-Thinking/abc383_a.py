def solve():
    n = int(input())
    additions = []
    for _ in range(n):
        t, v = map(int, input().split())
        additions.append((t, v))

    water = 0
    current_time = 0

    for t, v in additions:
        time_elapsed = t - current_time
        water_lost = time_elapsed
        water = max(0, water - water_lost)
        water += v
        current_time = t

    print(water)

solve()