MOD = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n-1]))
    
    max_val = 1000
    is_prime = [True] * (max_val+1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, max_val+1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_val+1, i):
                is_prime[j] = False
                
    ans = 1
    for p in primes:
        expo = []
        for num in A:
            cnt = 0
            x = num
            while x % p == 0:
                cnt += 1
                x //= p
            expo.append(cnt)
        
        I = []
        for i in range(len(expo)):
            if expo[i] > 0:
                I.append(i)
        m = len(I)
        if m == 0:
            continue
            
        candidate_list = [1]
        for i in I:
            candidate_list.append(i+1)
        candidate_list.append(n)
        candidate_list = sorted(set(candidate_list))
        
        total_assign = 0
        for bit in range(1<<m):
            assignment = {}
            for j in range(m):
                edge_index = I[j]
                assignment[edge_index] = (bit >> j) & 1
                
            C_val = 0
            for i in I:
                ci = assignment[i]
                C_val += (1 - ci) * expo[i]
                
            total_E_val = n * C_val
            for i in I:
                ci = assignment[i]
                total_E_val += (2*ci - 1) * expo[i] * (n - i - 1)
                
            E_vals = []
            for k in candidate_list:
                add_val = 0
                for i in I:
                    if i < k:
                        add_val += (2*assignment[i] - 1) * expo[i]
                E_val = C_val + add_val
                E_vals.append(E_val)
            min_E = min(E_vals)
            
            net_exp = total_E_val - n * min_E
            
            if net_exp >= 0:
                term = pow(p, net_exp, MOD)
            else:
                pos_exp = -net_exp
                term = pow(p, pos_exp, MOD)
                term = pow(term, MOD-2, MOD)
                
            multiplier = pow(2, (n-1) - m, MOD)
            total_assign = (total_assign + term * multiplier) % MOD
            
        ans = (ans * total_assign) % MOD
        
    print(ans)

if __name__ == '__main__':
    main()