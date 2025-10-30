def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append((l - 1, r - 1, x - 1))

    mod = 998244353

    def check_permutation(p):
        for l, r, x in conditions:
            sub_array = p[l : r + 1]
            if not sub_array:
                continue
            max_val = -1
            max_idx = -1
            for i in range(len(sub_array)):
                if sub_array[i] > max_val:
                    max_val = sub_array[i]
                    max_idx = i

            if l + max_idx == x:
                return False
        return True

    import itertools

    count = 0
    for p in itertools.permutations(range(1, n + 1)):
        if check_permutation(list(p)):
            count = (count + 1) % mod
    print(count)

solve()