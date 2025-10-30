def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    ans = 0
    from itertools import combinations

    for comb in combinations(a, k):
        xor_val = 0
        for val in comb:
            xor_val ^= val
        ans = max(ans, xor_val)

    print(ans)

solve()