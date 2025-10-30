def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    MOD = 998244353
    
    ans = 0
    
    import itertools
    
    for p in itertools.permutations(range(1, n + 1)):
        valid = True
        for i in range(n):
            for j in range(a[i] + 1, i + 1):
                if p[j-1] < p[i]:
                    valid = False
                    break
            if not valid:
                break
            if a[i] > 0 and p[a[i]-1] > p[i]:
                valid = False
                break
        if valid:
            ans = (ans + 1) % MOD
            
    print(ans)

solve()