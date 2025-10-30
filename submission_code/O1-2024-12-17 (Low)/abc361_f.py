def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip()
    N = int(input_data)
    
    # ------------------------------------------------------------------------
    # We wish to count how many integers x in [1..N] can be written as x=a^b
    # with a >= 1 and b >= 2.
    #
    # A classic high-level approach is:
    #   Count( x = a^2 ) + Count( x = a^3 ) + ... - (remove duplicates)
    # but N can be as large as 10^18, making direct enumeration impractical.
    #
    # Instead, note that any "perfect power" with exponent b>=2 has at least
    # one factorization of b into prime factors.  Equivalently, x=a^b = (a^d)^(b/d)
    # if d divides b.  Hence x belongs to the union:
    #       P_2 ∪ P_3 ∪ P_5 ∪ P_7 ∪ ... ∪ P_{prime≤59}
    # because the maximum exponent b for 2^b ≤ 10^18 is at most 59.
    #
    # By the principle of inclusion-exclusion:
    #
    #   |Union of P_p over primes p| =
    #       sum_{p} ( count of p-powers )
    #     - sum_{p1<p2} ( count of lcm(p1,p2)-powers )
    #     + sum_{p1<p2<p3} ( count of lcm(p1,p2,p3)-powers )
    #     - ...
    #
    # where "count of k-powers" = number of integers >=2 such that x=a^k ≤ N.
    # That count is floor(N^(1/k)) - 1 if that result is at least 1, else 0.
    #
    # We only consider subsets of primes whose LCM <= 59, because any larger LCM
    # would produce exponent > 59, and 2^60 > 10^18, so there would be no base ≥ 2
    # fitting a^LCM ≤ N.
    #
    # Finally, we add +1 if N >= 1 (to count x = 1, which is 1^b for all b≥2).
    #
    # Steps to implement:
    #  1) Generate all relevant primes up to 59.
    #  2) Use backtracking to list all non-empty subsets of these primes where
    #     the LCM of the subset is ≤ 59.  Record (lcm, subset_size) in a list.
    #  3) For each (lcm, sz), use inclusion-exclusion sign = (+1 if sz is odd, -1 if even).
    #     Add sign * (number of integers ≥2 that are perfect lcm-powers).
    #     The number of such integers is floor(N^(1/lcm)) - 1 if ≥ 1, else 0.
    #  4) Sum all contributions, then if N≥1, add 1 for "1" itself.
    #
    #  We implement floor(N^(1/k)) with an integer binary-search root finder
    #  to avoid floating precision issues.
    # ------------------------------------------------------------------------

    # Precompute all primes up to 59
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

    # Function to compute gcd
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    
    # Function to compute lcm(a,b) safely
    def lcm(a, b):
        return a // gcd(a, b) * b

    # We collect (lcm_val, size_of_subset) for all non-empty subsets with LCM <= 59
    lcm_subsets = []

    def backtrack(start_idx, current_lcm, depth):
        # We will try to include primes[start_idx..] one by one
        for i in range(start_idx, len(primes)):
            new_l = lcm(current_lcm, primes[i])
            if new_l > 59:
                continue
            # We have a new subset
            lcm_subsets.append((new_l, depth+1))
            backtrack(i+1, new_l, depth+1)

    # start with an empty subset => current_lcm=1, depth=0
    backtrack(0, 1, 0)

    # Function to compute floor of the k-th root of n, for k >= 1
    # via binary search, ensuring we don't rely on floating precision.
    def kth_root(n, k):
        # If n < 2, the only integer root is n itself.
        # But in our usage, N can be large, and we want the largest integer r >= 0
        # such that r^k <= n.
        if n < 2:
            return n
        left, right = 1, 10**18  # overkill upper bound
        # We'll refine a smaller upper bound. For 10^18, sqrt(10^18)=10^9 is plenty:
        # but for k > 1, we can do smaller, yet let's keep it simple.
        # We only do k <= 59, so 10^9 is safe enough for squares. We'll do something safe.
        # In practice, 10^9 for squares, 10^6 for cubes, etc. But let's just do a single approach:
        if k == 1:
            return n  # trivial
        # Binary search
        while left < right:
            mid = (left + right + 1) >> 1  # upper mid
            # Compute mid^k, but short-circuit if it exceeds n
            # We'll do repeated multiplication to avoid huge intermediate.
            val = 1
            for _ in range(k):
                val *= mid
                if val > n:
                    break
            if val <= n:
                left = mid
            else:
                right = mid - 1
        return left

    # Apply inclusion-exclusion
    total_count = 0
    for lval, size_ in lcm_subsets:
        # sign = +1 if subset-size is odd, -1 if even
        sign = 1 if (size_ % 2) == 1 else -1
        r = kth_root(N, lval)
        # We only count bases >= 2, so the count is max(0, r - 1)
        if r >= 2:
            total_count += sign * (r - 1)

    # Finally, add +1 for x=1 if N >= 1
    if N >= 1:
        total_count += 1

    print(total_count)

# Do not forget to call main()
main()