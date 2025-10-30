MOD = 998244353

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = 0
    
    import itertools
    for p in itertools.permutations(range(1, n + 1)):
        valid = True
        for i in range(n):
            for j in range(a[i], i):
                if p[j] <= p[i]:
                    valid = False
                    break
            if a[i] > 0 and p[a[i]-1] >= p[i]:
                valid = False
                break
            if not valid:
                break
        if valid:
            count += 1

    print(count % MOD)

solve()