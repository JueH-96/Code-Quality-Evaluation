def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    total_outcomes = 1
    for x in a:
        total_outcomes = (total_outcomes * x) % mod
    
    favorable_outcomes = 0
    
    import itertools
    
    for outcomes in itertools.product(*[range(1, x + 1) for x in a]):
        
        found_sum_10 = False
        for r in range(1, n + 1):
            for subset in itertools.combinations(range(n), r):
                subset_sum = 0
                for i in subset:
                    subset_sum += outcomes[i]
                if subset_sum == 10:
                    found_sum_10 = True
                    break
            if found_sum_10:
                break
        
        if found_sum_10:
            favorable_outcomes = (favorable_outcomes + 1) % mod
            
    
    if favorable_outcomes == 0:
        print(0)
        return
    
    
    def mod_inv(a, m):
        return pow(a, m - 2, m)
    
    
    probability = (favorable_outcomes * mod_inv(total_outcomes, mod)) % mod
    print(probability)

solve()