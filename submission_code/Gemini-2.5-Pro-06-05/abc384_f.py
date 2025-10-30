# YOUR CODE HERE
import sys

def main():
    """
    Main execution function.
    """
    # Fast I/O
    try:
        N = int(sys.stdin.readline())
        if N > 0:
            A = list(map(int, sys.stdin.readline().strip().split()))
        else: # N=0 case, though constraints say N>=1
            A = []
    except (IOError, ValueError):
        # This part handles potential empty input or format errors,
        # but problem constraints should prevent this.
        # Fallback for empty input.
        print(0)
        return

    # The function f(x) gives the largest odd divisor of x.
    # We want to compute Sum_{i <= j} f(A_i + A_j).
    # This is equivalent to ( Sum_{i,j} f(A_i + A_j) + Sum_{i} f(2*A_i) ) / 2.
    # Since f(2*x) = f(x), this simplifies to ( Sum_{i,j} f(A_i + A_j) + Sum_{i} f(A_i) ) / 2.
    # Let T = Sum_{i,j} f(A_i + A_j) and D = Sum_{i} f(A_i). The answer is (T + D) / 2.
    
    # Maximum possible v_2(A_i) for A_i <= 10^7 is floor(log2(10^7)) ~ 23.
    # We use 25 as a safe upper bound for the exponent of 2.
    MAX_V = 25 

    # D = sum_{i=1 to N} f(A_i)
    D = 0
    # group_by_v[v] stores the odd parts (B_i) of numbers A_i where v_2(A_i) = v
    group_by_v = [[] for _ in range(MAX_V)]

    for x in A:
        if x == 0: continue
        # v_2(x) can be found by looking at the lowest set bit.
        low_bit = x & -x
        v = low_bit.bit_length() - 1
        b = x >> v  # b is the odd part, which is f(x)
        
        D += b
        if v < MAX_V:
            group_by_v[v].append(b)

    # T = sum_{i,j} f(A_i + A_j)
    T = 0

    # Precompute counts (n_v) and sums of odd parts (SB_v) for each group.
    n = [0] * MAX_V
    SB = [0] * MAX_V
    for v in range(MAX_V):
        n[v] = len(group_by_v[v])
        if n[v] > 0:
            SB[v] = sum(group_by_v[v])

    # Case 1: v_2(A_i) != v_2(A_j)
    for u in range(MAX_V):
        if n[u] == 0: continue
        for v in range(u + 1, MAX_V):
            if n[v] == 0: continue
            # Contribution from pairs (i, j) where i is in group u, j is in group v.
            # f(A_i+A_j) is (A_i+A_j)/2^u if u < v.
            # Sum over these pairs is n_v * SB_u + n_u * 2^(v-u) * SB_v.
            # We add this for pairs (i in I_u, j in I_v) and (i in I_v, j in I_u).
            term = n[v] * SB[u] + n[u] * (1 << (v - u)) * SB[v]
            T += 2 * term

    # Case 2: v_2(A_i) == v_2(A_j)
    def calc_sub(C):
        n_c = len(C)
        if n_c == 0:
            return 0
        
        max_b = 0
        for b_val in C:
            if b_val > max_b:
                max_b = b_val
        
        max_k = (2 * max_b).bit_length() if max_b > 0 else 0
        
        S_k_plus_1 = 0
        total_sub_sum = 0
        
        for k in range(max_k, 0, -1):
            mod = 1 << k
            
            counts = {}
            sums = {}
            for b in C:
                r = b % mod
                counts[r] = counts.get(r, 0) + 1
                sums[r] = sums.get(r, 0) + b
            
            current_S_k = 0
            processed_r = set()
            for r in counts:
                if r in processed_r:
                    continue
                
                r_comp = (mod - r) % mod
                
                if r == r_comp:
                    current_S_k += 2 * counts[r] * sums[r]
                elif r_comp in counts:
                    current_S_k += counts[r_comp] * sums[r] + counts[r] * sums[r_comp]
                    processed_r.add(r_comp)
                
                processed_r.add(r)
            
            S_k = current_S_k
            Term_k = S_k - S_k_plus_1
            total_sub_sum += Term_k // mod
            S_k_plus_1 = S_k
            
        return total_sub_sum

    for v in range(MAX_V):
        if n[v] > 0:
            T += calc_sub(group_by_v[v])
            
    ans = (T + D) // 2
    print(ans)

if __name__ == "__main__":
    main()