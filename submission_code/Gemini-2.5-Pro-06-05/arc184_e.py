import sys

# It is necessary to increase the recursion limit for the recursive solution.
sys.setrecursionlimit(2 * 10**6)

def solve():
    """
    Main solver function for the problem.
    """
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353
    
    # We group sequences 'B' by the index of their first '1'.
    # B is the difference sequence of A, i.e., B = D(A).
    # f(A_i, A_j) is equivalent to finding x for T^x(B_i) = B_j
    # This can be non-zero only if the first '1' in B_i and B_j are at the same position.
    b_seqs_by_k = {}

    for _ in range(N):
        try:
            a_str = sys.stdin.readline().split()
            if not a_str: continue
            a = [int(x) for x in a_str]
        except (IOError, ValueError):
            continue
        
        b = [0] * M
        b[0] = a[0]
        for i in range(1, M):
            b[i] = (a[i-1] + a[i]) % 2
        
        # Find the 1-based index of the first '1' in B.
        k = -1
        for i in range(M):
            if b[i] == 1:
                k = i + 1
                break
        
        # Group B sequences by k. If B is all zeros, k remains -1.
        # We only need the part of the sequence starting from the first '1'.
        if k != -1:
            b_shifted = tuple(b[k-1:])
            if k not in b_seqs_by_k:
                b_seqs_by_k[k] = []
            b_seqs_by_k[k].append(b_shifted)

    total_sum = 0
    
    for k in b_seqs_by_k:
        shifted_b_list = b_seqs_by_k[k]
        
        # For each shifted sequence B', find its canonical form (e, Q).
        # B' = D^e(Q), where Q is the "base" sequence for the orbit.
        # e is the smallest non-negative integer such that the sum of elements in T^e(B') is odd.
        # Q = T^e(B').
        
        # Group sequences by their canonical base sequence Q.
        q_groups = {}
        
        # Length of the shifted sequences.
        L = M - k + 1

        for b_prime in shifted_b_list:
            b_curr = list(b_prime)
            e = 0
            
            # The sum of elements in b_curr will eventually become odd.
            # b_prime[0] is 1, so T^e(b_prime) can't be all zeros.
            # The sequence of popcounts is periodic, so this loop terminates.
            while sum(b_curr) % 2 == 0:
                e += 1
                
                # Apply transformation T (prefix sum mod 2)
                b_next = [0] * L
                current_sum = 0
                for i in range(L):
                    current_sum = (current_sum + b_curr[i]) % 2
                    b_next[i] = current_sum
                
                b_curr = b_next

            q = tuple(b_curr)
            if q not in q_groups:
                q_groups[q] = []
            q_groups[q].append(e)

        # For each Q, we have a list of 'e' values.
        # g(B'_i, B'_j) = e_i - e_j if Q_i == Q_j and e_i >= e_j, else 0.
        # We need to calculate sum_{e_a > e_b} c(e_a)c(e_b)(e_a-e_b) for all pairs.
        for q in q_groups:
            e_list = q_groups[q]
            
            e_counts = {}
            for e_val in e_list:
                e_counts[e_val] = e_counts.get(e_val, 0) + 1
            
            sorted_e = sorted(e_counts.keys())
            
            term_sum = 0
            count_prefix_sum = 0
            weighted_count_prefix_sum = 0
            
            for e_val in sorted_e:
                count = e_counts[e_val]
                
                # Contribution from pairs where current e_val is the larger one.
                # (e_val * sum_{j<i} c_j) - (sum_{j<i} c_j e_j)
                term_sum = (term_sum + count * ((e_val * count_prefix_sum) % MOD)) % MOD
                term_sum = (term_sum - (count * weighted_count_prefix_sum) % MOD + MOD) % MOD
                
                # Update prefix sums
                count_prefix_sum = (count_prefix_sum + count)
                weighted_count_prefix_sum = (weighted_count_prefix_sum + (count * e_val))

            total_sum = (total_sum + term_sum) % MOD

    print(total_sum)

solve()