def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    
    # A small helper: how many times 2 divides x
    def factor2(x):
        cnt = 0
        while x % 2 == 0:
            x //= 2
            cnt += 1
        return cnt

    # Precompute minimal increments to add +1 or +2 "powers of 2" to each number up to 10,
    # but we may need to go above 10 by increments. We'll allow up to +16 increments
    # which covers up to 26. That is plenty to get at least +3 or +4 powers of 2 if needed.
    # We'll create a table cost_plus[i][extra] = minimal increments needed so that
    # factors2(i + d) >= factors2(i) + extra.
    # We only ever need extra = 1 or 2 for k=4. (Because we want up to 2 additional factors of 2.)
    MAX_INCR = 16

    # Instead of just for i in [1..10], we might do for i in [0..30], but since a_i>=1..10,
    # let's do for i in range(1..10+MAX_INCR). Some numbers up to 10+16=26, so do up to 26 or 30 for safe margin.
    MAX_BASE = 30
    # cost_plus[x][extra] will be defined for x in [1..MAX_BASE], extra in [0..2].
    # If x>MAX_BASE (like 27,28,... after increments) we won't need that because we only do increments up to 16 from base up to 10 + 16=26.
    # So let's fill it with None initially, then fill.
    cost_plus = [[10**9]*3 for _ in range(MAX_BASE+1)]
    
    for x in range(1, MAX_BASE+1):
        base_factor2 = factor2(x)
        # cost_plus[x][0] = 0 (no extra factors needed => 0 increments)
        cost_plus[x][0] = 0
        # we try d from 0..MAX_INCR
        for d in range(MAX_INCR+1):
            y = x + d
            if y <= MAX_BASE:
                new_f2 = factor2(y)
                gained = new_f2 - base_factor2
                # if gained >= 1, update cost_plus[x][1]
                if gained >= 1:
                    cost_plus[x][1] = min(cost_plus[x][1], d)
                # if gained >= 2, update cost_plus[x][2]
                if gained >= 2:
                    cost_plus[x][2] = min(cost_plus[x][2], d)

    idx = 1
    out = []
    for _ in range(t):
        n = int(input_data[idx]); idx+=1
        k = int(input_data[idx]); idx+=1
        
        arr = list(map(int, input_data[idx:idx+n]))
        idx += n
        
        # Handle each k separately
        if k == 2:
            # We need at least one factor of 2 in the product => at least one even number
            found_even = any((x % 2 == 0) for x in arr)
            if found_even:
                out.append('0')
            else:
                # we need to increment one odd number by 1 => cost=1
                # minimal cost is always 1, because from any odd x to next even is +1
                out.append('1')
        
        elif k == 3:
            # We need at least one multiple of 3
            found_mult3 = any((x % 3 == 0) for x in arr)
            if found_mult3:
                out.append('0')
            else:
                # pick the minimum cost to become divisible by 3: min( (3 - x%3) )
                best = min((3 - (x % 3)) for x in arr)
                out.append(str(best))
        
        elif k == 5:
            # We need at least one multiple of 5
            found_mult5 = any((x % 5 == 0) for x in arr)
            if found_mult5:
                out.append('0')
            else:
                # pick the minimum cost to become divisible by 5: min( (5 - x%5) )
                best = min((5 - (x % 5)) for x in arr)
                out.append(str(best))
        
        else:
            # k == 4 => we need at least total 2 factors of 2 in the product
            # Count how many 2-factors we have initially
            total_2 = 0
            for x in arr:
                total_2 += factor2(x)
            if total_2 >= 2:
                out.append('0')
            else:
                # We'll gather cost_1[i], cost_2[i] = cost to add +1 or +2 factors of 2 beyond current
                # factor2(x) = e, we want factors2(x+d) >= e+1 or >= e+2.
                # We can retrieve from cost_plus[x].
                # But if x>MAX_BASE, we can clamp it to MAX_BASE because arr[i] <= 10, so it's safe.
                
                # cost_1[i] = cost to add exactly +1 factor beyond current (if possible),
                # or 10^9 if we already have that many or it's not feasible. But we'll use cost_plus.
                from math import inf
                cost_1 = []
                cost_2 = []
                
                # Also track each x's current factor2
                arr_f2 = [factor2(x) for x in arr]
                for i, x in enumerate(arr):
                    base = x
                    if base > MAX_BASE:
                        base = MAX_BASE
                    e = arr_f2[i]
                    
                    # We want the cost to get factors2 >= e+1
                    # but cost_plus[base][1] always tries to move from factor2(base) to factor2(base+ d)... 
                    # We must check if factor2(base+ d) >= e + 1, but cost_plus uses the difference from factor2(base).
                    # That difference is factor2(x + d) - factor2(x).
                    # If that difference >= 1, cost_plus[x][1] is minimal d for that difference >= 1.
                    # But if the current factor2(x) is e, we want e + 1 total.  The difference needed is 1 if e=0,
                    # or if e=1, the difference needed is 0 => we do cost_plus[x][0]? Actually we want "one more factor" 
                    # than it already has. If x already has factor2=1, we need factor2(x+d) >= 2 => difference >=1 again.
                    # So cost_plus[x][1] is exactly "the minimal increments needed to gain +1 factor of 2 beyond the current exponent".
                    
                    # Similarly cost_plus[x][2] is "the minimal increments needed to gain +2 beyond the current exponent."
                    
                    # But if e >= 1, we only want one more factor, that means difference >= 1 from e, 
                    # i.e. factor2 => e+1 => difference = (e+1) - e = 1 => cost_plus[x][1].
                    # If e >= 2, we do not need to pay anything for cost_2 or cost_1 because we already have that many factors.
                    
                    # So effectively if e >= 2 => cost_1=0, cost_2=0
                    # elif e=1 => cost_1=0, cost_2 = cost_plus[x][1] (the cost to go from difference=1 to difference=2 overall => actually we want +1 from e=1 => that means difference=2 from factor_2(x)? We must be careful!)
                    # let's do a clearer approach:
                    
                    # number_of_factors_needed_for_1_more = (arr_f2[i] + 1) - arr_f2[i] = 1
                    # number_of_factors_needed_for_2_more = (arr_f2[i] + 2) - arr_f2[i] = 2
                    # but if arr_f2[i] >= 2 => cost_1 = cost_2 = 0
                    # if arr_f2[i] = 1 => cost_1=0, cost_2 = cost_plus[x][1]
                    # if arr_f2[i] = 0 => cost_1 = cost_plus[x][1], cost_2 = cost_plus[x][2]
                    
                    e = arr_f2[i]
                    if e >= 2:
                        c1 = 0
                        c2 = 0
                    elif e == 1:
                        # no cost to get +1 because we already have factor2 >=1
                        c1 = 0
                        c2 = cost_plus[base][1]  # cost to go from difference=1 to difference=2 => i.e. +1 more factor
                    else:
                        # e = 0
                        c1 = cost_plus[base][1]
                        c2 = cost_plus[base][2]
                    
                    cost_1.append(c1)
                    cost_2.append(c2)
                
                # now we have total_2 = 0 or 1.
                # if total_2 = 1 => we need 1 more factor => cost = min of cost_1[i]
                if total_2 == 1:
                    ans = min(cost_1)
                    out.append(str(ans))
                else:
                    # total_2 = 0 => we need 2 factors
                    # option A: pick one element i to add 2 factors => cost_2[i]
                    A = min(cost_2)
                    # option B: pick two distinct elements i, j to add 1 factor each => cost_1[i] + cost_1[j]
                    # but we only need elements that can indeed add +1 factor. 
                    # The cost_1 might be 0 for an element that already has factor_2 >=1, that helps. 
                    # We'll just sort cost_1 and take the sum of the two smallest.
                    sorted_c1 = sorted(cost_1)
                    if len(sorted_c1) >= 2:
                        B = sorted_c1[0] + sorted_c1[1]
                    else:
                        B = 10**9
                    ans = min(A,B)
                    out.append(str(ans))
    
    print('
'.join(out))

# Remember to call main()
if __name__ == "__main__":
    main()