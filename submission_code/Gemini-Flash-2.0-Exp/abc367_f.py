def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        l, r, L, R = map(int, input().split())
        l -= 1
        L -= 1
        sub_a = a[l:r]
        sub_b = b[L:R]

        if len(sub_a) != len(sub_b):
            print("No")
            continue

        counts_a = {}
        for x in sub_a:
            counts_a[x] = counts_a.get(x, 0) + 1

        counts_b = {}
        for x in sub_b:
            counts_b[x] = counts_b.get(x, 0) + 1

        possible = True
        for x in counts_a:
            if x not in counts_b or counts_a[x] != counts_b[x]:
                possible = False
                break
        
        if possible:
            for x in counts_b:
                if x not in counts_a or counts_b[x] != counts_a[x]:
                    possible = False
                    break

        if possible:
            print("Yes")
        else:
            print("No")

solve()