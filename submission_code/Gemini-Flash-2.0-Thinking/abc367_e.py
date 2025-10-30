def solve():
    n, k = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    if k == 0:
        print(*a)
        return

    p = [val - 1 for val in x]

    def compose(p1, p2):
        n = len(p1)
        result = [0] * n
        for i in range(n):
            result[i] = p1[p2[i]]
        return result

    def power_permutation(perm, k):
        n = len(perm)
        identity = list(range(n))
        result = identity
        base = perm
        while k > 0:
            if k % 2 == 1:
                result = compose(result, base)
            base = compose(base, base)
            k //= 2
        return result

    p_k = power_permutation(p, k)
    final_a = [0] * n
    for i in range(n):
        final_a[i] = a[p_k[i]]

    print(*final_a)

solve()