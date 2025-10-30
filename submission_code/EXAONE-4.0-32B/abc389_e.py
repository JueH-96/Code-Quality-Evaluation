import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:2+n]))
    
    if n == 0:
        print(0)
        return
        
    freq = {}
    for p in P:
        q = 2 * p
        freq[q] = freq.get(q, 0) + 1
    sorted_q = sorted(freq.keys())
    
    low_T = 0
    high_T = 10**18
    ans = 0
    
    while low_T <= high_T:
        mid_T = (low_T + high_T) // 2
        if mid_T == 0:
            cost_val = 0
            if cost_val <= M:
                ans = mid_T
                low_T = mid_T + 1
            else:
                high_T = mid_T - 1
            continue
        
        low_lam = 1
        high_lam = 10**15
        lambda0 = None
        while low_lam <= high_lam:
            mid_lam = (low_lam + high_lam) // 2
            X = mid_lam - 1
            if X < 1:
                g_val = 0
            else:
                g_val = 0
                for q in sorted_q:
                    if q > X:
                        break
                    g_val += freq[q] * (X // q)
            if g_val >= mid_T:
                lambda0 = mid_lam
                high_lam = mid_lam - 1
            else:
                low_lam = mid_lam + 1
                
        if lambda0 is None:
            cost_val = 10**30
        else:
            base_k = []
            S0 = 0
            for p in P:
                num = lambda0 - 2
                if num < 0:
                    k_i = 0
                else:
                    k_i = num // (2 * p)
                base_k.append(k_i)
                S0 += k_i
            R = mid_T - S0
            candidate_list = []
            for i in range(len(P)):
                p = P[i]
                k_i_val = base_k[i]
                if (k_i_val + 1) * 2 * p <= lambda0 - 1:
                    marginal_cost = (2 * k_i_val + 1) * p
                    candidate_list.append(marginal_cost)
            candidate_list.sort()
            add_cost = sum(candidate_list[:R]) if R <= len(candidate_list) else sum(candidate_list)
            base_cost = 0
            for i in range(len(P)):
                base_cost += P[i] * base_k[i] * base_k[i]
            cost_val = base_cost + add_cost
            
        if cost_val <= M:
            ans = mid_T
            low_T = mid_T + 1
        else:
            high_T = mid_T - 1
            
    print(ans)

if __name__ == "__main__":
    main()