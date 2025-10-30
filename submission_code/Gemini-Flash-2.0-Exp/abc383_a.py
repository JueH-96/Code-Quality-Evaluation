def solve():
    n = int(input())
    t = []
    v = []
    for _ in range(n):
        ti, vi = map(int, input().split())
        t.append(ti)
        v.append(vi)

    water = 0
    last_time = 0
    
    for i in range(n):
        time_diff = t[i] - last_time
        water = max(0, water - time_diff)
        water += v[i]
        last_time = t[i]

    print(water)

solve()