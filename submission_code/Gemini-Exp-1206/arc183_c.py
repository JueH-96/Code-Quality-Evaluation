def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        conditions.append(list(map(int, input().split())))

    mod = 998244353

    def check_permutation(perm):
        for l, r, x in conditions:
            max_val = 0
            max_idx = -1
            for i in range(l - 1, r):
                if perm[i] > max_val:
                    max_val = perm[i]
                    max_idx = i
            if max_idx == x - 1:
                return False
        return True

    import itertools

    count = 0
    for perm_tuple in itertools.permutations(range(1, n + 1)):
        perm = list(perm_tuple)
        if check_permutation(perm):
            count += 1

    print(count % mod)

solve()