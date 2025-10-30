def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(list(map(int, input().split())))

    ans = 0
    import itertools
    for p in itertools.permutations(dishes):
        cur_sweet = 0
        cur_salt = 0
        count = 0
        for dish in p:
            cur_sweet += dish[0]
            cur_salt += dish[1]
            if cur_sweet > x or cur_salt > y:
                break
            count += 1
        ans = max(ans, count)

    print(ans)

solve()