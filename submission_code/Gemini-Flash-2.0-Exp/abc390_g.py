def solve():
    n = int(input())
    a = list(range(1, n + 1))
    
    import itertools
    
    total_sum = 0
    mod = 998244353
    
    for p in itertools.permutations(a):
        s = ""
        for x in p:
            s += str(x)
        total_sum = (total_sum + int(s)) % mod
        
    print(total_sum)

solve()