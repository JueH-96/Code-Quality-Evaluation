def solve():
    n, m = map(int, input().split())
    s = input()

    def can_satisfy(n, m, s, k):
        plain = m
        logo = k
        for day_schedule in s:
            if day_schedule == '0':
                plain = m + k
                logo = k
            elif day_schedule == '1':
                if plain > 0:
                    plain -= 1
                elif logo > 0:
                    logo -= 1
                else:
                    return False
            elif day_schedule == '2':
                if logo > 0:
                    logo -= 1
                else:
                    return False
        return True

    for k in range(n + 1):
        if can_satisfy(n, m, s, k):
            print(k)
            return

solve()