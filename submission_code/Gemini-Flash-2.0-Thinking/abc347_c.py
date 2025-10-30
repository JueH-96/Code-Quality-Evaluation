def solve():
    n, a, b = map(int, input().split())
    d = list(map(int, input().split()))
    w = a + b

    for h1 in range(1, a + 1):
        start_day_mod_w = (h1 - d[0]) % w
        if start_day_mod_w < 0:
            start_day_mod_w += w

        possible = True
        for plan_day in d:
            day_of_week = (start_day_mod_w + plan_day) % w
            if not (0 <= day_of_week <= a - 1):
                possible = False
                break
        if possible:
            print("Yes")
            return

    print("No")

solve()