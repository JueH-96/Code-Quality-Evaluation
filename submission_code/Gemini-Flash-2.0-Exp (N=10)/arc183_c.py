def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append((l - 1, r - 1, x - 1))

    def check_permutation(p):
        for l, r, x in conditions:
            max_val = -1
            max_idx = -1
            for i in range(l, r + 1):
                if p[i] > max_val:
                    max_val = p[i]
                    max_idx = i
            if max_idx == x:
                return False
        return True

    import itertools
    count = 0
    for p in itertools.permutations(range(1, n + 1)):
        if check_permutation(list(p)):
            count += 1
    print(count % 998244353)

solve()