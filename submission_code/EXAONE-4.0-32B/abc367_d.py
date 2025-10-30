import sys
from collections import defaultdict
import math
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
        
    n = int(data[0])
    M_val = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    P = [0] * (n+1)
    for i in range(1, n+1):
        P[i] = P[i-1] + A[i-1]
    S_total = P[n]
    
    if S_total == 0:
        print(0)
        return
        
    g = math.gcd(M_val, S_total)
    M_prime = M_val // g
    S_prime = S_total // g
    
    B = P[:n]
    
    groups = defaultdict(list)
    for x in B:
        r = x % g
        groups[r].append(x)
        
    if S_prime == 0:
        print(0)
        return
        
    L_prime = (S_prime - 1) // M_prime
    if L_prime <= 0:
        print(0)
        return
        
    total_pairs = 0
    for r, arr in groups.items():
        T_arr = []
        for x in arr:
            T_arr.append((x - r) // g)
            
        T_mod = [t % S_prime for t in T_arr]
        
        try:
            inv_mp = pow(M_prime, -1, S_prime)
        except:
            inv_mp = 1
            
        U = [ (t * inv_mp) % S_prime for t in T_mod ]
        
        V = U[:]
        for u_val in U:
            V.append(u_val + S_prime)
            
        V.sort()
        
        for u_val in U:
            low_bound = u_val + 1
            high_bound = u_val + L_prime
            left_index = bisect.bisect_left(V, low_bound)
            right_index = bisect.bisect_right(V, high_bound)
            count_here = right_index - left_index
            total_pairs += count_here
            
    print(total_pairs)

if __name__ == '__main__':
    main()