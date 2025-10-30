def solve():
    n, q = map(int, input().split())
    h = list(map(int, input().split()))
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    for l_q, r_q in queries:
        l = l_q - 1
        r = r_q - 1
        count = 0
        for k in range(r + 1, n):
            # Check visibility from l to k
            can_see_from_l = True
            for m in range(l + 1, k):
                if h[m] > h[k]:
                    can_see_from_l = False
                    break

            # Check visibility from r to k
            can_see_from_r = True
            for m in range(r + 1, k):
                if h[m] > h[k]:
                    can_see_from_r = False
                    break

            if can_see_from_l and can_see_from_r:
                count += 1
        print(count)

solve()