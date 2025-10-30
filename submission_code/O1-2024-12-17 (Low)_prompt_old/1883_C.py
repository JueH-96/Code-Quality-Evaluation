def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    ptr = 1
    
    # Precompute the factor of 2 for numbers up to a bit above 20
    # (since we only need up to exponent 2 for k=4, going up to 20 is enough)
    # We'll store exponentOf2(x) in a small table for x up to 30 to be safe.
    MAX_CHECK = 30
    exp2 = [0]*(MAX_CHECK+1)
    for x in range(1, MAX_CHECK+1):
        # count how many times x divides by 2
        tmp = x
        cnt = 0
        while tmp % 2 == 0:
            tmp //= 2
            cnt += 1
        exp2[x] = cnt
    
    # A helper to get cost to raise exponentOf2(a) up to at least targetExp
    # We'll only use targetExp <= 2, and a <= 10 + a few increments
    # We search x in [0..12] (increment up to a+12) to find minimal cost
    def cost_to_get_exp2(a, current_exp, target_exp):
        if current_exp >= target_exp:
            return 0
        best = float('inf')
        # We'll only check up to 20 or so in total
        for inc in range(12+1):
            val = a + inc
            if val <= MAX_CHECK and exp2[val] >= target_exp:
                best = min(best, inc)
        return best
    
    out = []
    for _ in range(t):
        n = int(input_data[ptr]); ptr+=1
        k = int(input_data[ptr]); ptr+=1
        arr = list(map(int, input_data[ptr:ptr+n]))
        ptr+=n
        
        # Handle prime k = 2,3,5 separately
        # Then handle k=4 with sum of exponents of 2 logic
        if k in [2, 3, 5]:
            # We need at least one multiple of k in the product
            # Check if there's already a multiple of k
            already = any(a % k == 0 for a in arr)
            if already:
                out.append('0')
            else:
                # We need the minimal increments to make at least one a_i divisible by k
                # cost = (k - (a_i % k)) % k for each a_i, we take min
                best = min((k - (x % k)) % k for x in arr)
                out.append(str(best))
        
        else:
            # k = 4 => we need at least two factors of 2 in the product
            # sum up the exponents of 2
            e2_list = [exp2[x] for x in arr]
            S = sum(e2_list)
            if S >= 2:
                out.append('0')
                continue
            
            # Precompute cost1 and cost2 for each element:
            # cost1[i]: minimal cost to increase exponentOf2 by at least +1 from current
            # cost2[i]: minimal cost to increase exponentOf2 to at least 2
            # If the current exponent is e_i, then cost1[i] = 0 if e_i >= 1, else the cost to get exponent >=1.
            # Similarly cost2[i] = 0 if e_i >= 2, etc.
            cost1 = []
            cost2 = []
            for a,e in zip(arr, e2_list):
                c1 = 0 if e >= 1 else cost_to_get_exp2(a, e, 1)
                c2 = 0 if e >= 2 else cost_to_get_exp2(a, e, 2)
                cost1.append(c1)
                cost2.append(c2)
            
            if S == 1:
                # We only need 1 more factor of 2
                # That is min of cost1 over all i
                ans = min(cost1)
                out.append(str(ans))
            else:
                # S == 0, we need 2 total factors
                # We can do either:
                # 1) Take min(cost2) for some i (one element going from 0 exponent to 2 exponent)
                # 2) Take sum of two distinct cost1's (two elements each from 0 exponent to 1 exponent)
                # We want the minimal of these approaches.
                
                # min cost2
                best_c2 = min(cost2)
                
                # min sum of two distinct cost1
                # sort cost1, take the two smallest
                cost1_sorted = sorted(cost1)
                if len(cost1_sorted) >= 2:
                    best_c1_c1 = cost1_sorted[0] + cost1_sorted[1]
                else:
                    # not possible to pick two distinct? (n>=2, so normally we can pick two)
                    best_c1_c1 = float('inf')
                
                ans = min(best_c2, best_c1_c1)
                out.append(str(ans))
    
    print('
'.join(out))