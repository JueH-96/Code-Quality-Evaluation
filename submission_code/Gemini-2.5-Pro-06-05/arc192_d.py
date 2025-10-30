import sys
from collections import defaultdict

def main():
    """
    This is the main function that reads input, solves the problem, and prints the output.
    The solution logic is encapsulated in the `solve` function.
    """
    
    # Encapsulating the solution within a `solve` function
    # to better structure the code and follow the requested format.
    def solve():
        """
        This function solves the problem by breaking it down prime by prime.
        For each prime p, we calculate the contribution to the total sum of scores.
        The final answer is the product of these contributions modulo 998244353.

        Let v_p(n) be the exponent of prime p in the factorization of n.
        Let e_i = v_p(S_i) and a_i = v_p(A_i).
        The condition f(S_i / S_{i+1}) = A_i implies that for each prime p,
        e_{i+1} = e_i + c_i * a_i, where c_i is either +1 or -1.
        Also, if c_i = -1, we must have e_i >= a_i.
        The condition gcd(S_1, ..., S_N) = 1 implies min(e_1, ..., e_N) = 0 for each p.

        For a fixed sequence of choices (c_1, ..., c_{N-1}), the sequence of exponents e_i is
        uniquely determined up to an additive constant. Let B_i = sum_{j=1}^{i-1} c_j * a_j (with B_1=0).
        Then e_i = e_1 + B_i. The condition min(e_i) = 0 implies e_1 = -min(B_k for k=1..N).
        So, e_i = B_i - min(B_k for k=1..N).

        The score contribution from prime p is p^{sum(e_i)}.
        We need to sum this over all valid choice sequences.
        A choice sequence is valid if e_i >= a_i whenever c_i = -1.

        We can partition the set of valid choice sequences by the first index k0 where B_k reaches its minimum.
        For such a path, let B'_j = B_j - B_{k0}. Then B'_{k0}=0, B'_j >= 0 for j > k0, and B'_j > 0 for j < k0.
        The exponent sequence is e_j = B'_j.
        The score contribution for this path is p^{sum(B'_j)}.
        The validity condition is B'_j >= a_j if c_j = -1.

        We can use dynamic programming. For each k0 from 1 to N:
        1. Forward DP: Calculate score sum for paths from k0 to N.
           Let F[b] be the sum of p^{sum_{j=k0 to i} B'_j} for paths from k0 to i ending at B'_i=b.
        2. Backward DP: Calculate score sum for paths from k0 to 1.
           Let B[b] be the sum of p^{sum_{j=i to k0} B'_j} for paths from i to k0 ending at B'_i=b,
           with the strict inequality constraint B'_j > 0 for j < k0.
        The total contribution for k0 is (sum over F) * (sum over B).
        Summing over all k0 gives the total for prime p.
        """
        try:
            input_stream = sys.stdin
            N_str = input_stream.readline()
            if not N_str: return
            N = int(N_str)

            if N == 1:
                print(1)
                return
            
            A = list(map(int, input_stream.readline().split()))
        except (IOError, ValueError):
            return

        MOD = 998244353

        prime_factors_map = defaultdict(int)
        for x in A:
            d = 2
            temp_x = x
            while d * d <= temp_x:
                if temp_x % d == 0:
                    prime_factors_map[d] = 1
                    while temp_x % d == 0:
                        temp_x //= d
                d += 1
            if temp_x > 1:
                prime_factors_map[temp_x] = 1
        
        total_score_sum = 1

        for p in prime_factors_map:
            a = []
            for x in A:
                count = 0
                temp_x = x
                while temp_x > 0 and temp_x % p == 0:
                    count += 1
                    temp_x //= p
                a.append(count)
            
            max_b_val = sum(a)
            
            p_powers = [1] * (max_b_val + 1)
            for i in range(1, len(p_powers)):
                p_powers[i] = (p_powers[i - 1] * p) % MOD

            total_p = 0
            
            for k0 in range(1, N + 1):
                # Forward DP from k0 to N
                F = {0: 1}
                for i in range(k0 - 1, N - 1):
                    F_next = defaultdict(int)
                    ai = a[i]
                    for b, val in F.items():
                        # Case c_i = +1
                        b_new = b + ai
                        if b_new <= max_b_val:
                            term = (val * p_powers[b_new]) % MOD
                            F_next[b_new] = (F_next[b_new] + term) % MOD
                        
                        # Case c_i = -1
                        if b >= ai:
                            b_new = b - ai
                            term = (val * p_powers[b_new]) % MOD
                            F_next[b_new] = (F_next[b_new] + term) % MOD
                    F = F_next
                
                sum_f = sum(F.values()) % MOD

                # Backward DP from k0 to 1
                sum_b = 1
                if k0 > 1:
                    B = {0: 1}
                    for i in range(k0, 1, -1):
                        B_next = defaultdict(int)
                        ai_minus_1 = a[i - 2]
                        for b, val in B.items():
                            if i < k0 and b == 0: continue

                            # Case c_{i-1} = +1: B'_{i-1} = B'_i - a_{i-1}
                            if b > ai_minus_1:
                                b_new = b - ai_minus_1
                                term = (val * p_powers[b_new]) % MOD
                                B_next[b_new] = (B_next[b_new] + term) % MOD
                            
                            # Case c_{i-1} = -1: B'_{i-1} = B'_i + a_{i-1}
                            if b >= 0:
                                b_new = b + ai_minus_1
                                if b_new > 0 and b_new <= max_b_val:
                                    term = (val * p_powers[b_new]) % MOD
                                    B_next[b_new] = (B_next[b_new] + term) % MOD
                        B = B_next
                    sum_b = sum(B.values()) % MOD

                total_p = (total_p + sum_f * sum_b) % MOD

            total_score_sum = (total_score_sum * total_p) % MOD
            
        print(total_score_sum)

    # Set a higher recursion limit for safety, although this solution is iterative.
    sys.setrecursionlimit(2000)
    # The `main` function calls the `solve` function to run the solution.
    solve()

if __name__ == "__main__":
    main()