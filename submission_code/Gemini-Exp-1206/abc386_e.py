def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    ans = 0
    
    if k == 1:
        ans = max(a)
    elif n == k:
        ans = 0
        for x in a:
            ans ^= x
    else:
        import itertools
        for comb in itertools.combinations(a, k):
            xor_sum = 0
            for x in comb:
                xor_sum ^= x
            ans = max(ans, xor_sum)

    print(ans)

solve()