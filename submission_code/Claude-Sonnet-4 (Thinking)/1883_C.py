def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k == 2:
        for x in a:
            if x % 2 == 0:
                return 0
        return 1
    
    elif k == 3:
        for x in a:
            if x % 3 == 0:
                return 0
        return min((3 - (x % 3)) % 3 for x in a)
    
    elif k == 4:
        total_factors_of_2 = 0
        for x in a:
            temp = x
            while temp % 2 == 0:
                total_factors_of_2 += 1
                temp //= 2
        
        if total_factors_of_2 >= 2:
            return 0
        
        min_to_make_div4 = min((4 - (x % 4)) % 4 for x in a)
        
        needed_factors = 2 - total_factors_of_2
        costs = []
        for x in a:
            temp = x
            factors = 0
            while temp % 2 == 0:
                factors += 1
                temp //= 2
            costs.append(2 ** factors)
        
        costs.sort()
        min_to_add_factors = sum(costs[:needed_factors])
        
        return min(min_to_make_div4, min_to_add_factors)
    
    elif k == 5:
        for x in a:
            if x % 5 == 0:
                return 0
        return min((5 - (x % 5)) % 5 for x in a)

t = int(input())
for _ in range(t):
    print(solve())