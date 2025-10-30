def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    # Read input
    N_str = sys.stdin.read().strip()
    N = int(N_str)
    
    # -------------------------------------------------------------------
    # We want to count how many integers x in [1..N] can be written as x=a^b
    # with a >= 1 and b >= 2.  (In particular, 1=1^b is included.)
    #
    # A classical observation is that any integer that is a perfect power
    # with exponent b >= 2 will also be a perfect power with exponent equal
    # to any prime factor of b.  For example, if x=a^(6), then x=(a^3)^2 is
    # also a perfect square.  Consequently, to find all perfect powers up
    # to N, it suffices to consider only prime exponents b ∈ [2..59].
    # (Because 2^59 is already > 10^18 in magnitude, 59 is more than enough.)
    #
    # However, in order to avoid double-counting numbers that are, say, both
    # perfect squares and perfect cubes (which are perfect 6th powers, etc.),
    # we do an inclusion-exclusion over sets PP'(p) = { x : x=a^p for some a>1 }
    # for each prime p in [2..59].  But we do not need to consider all p ∈ [2..59]
    # individually if p is composite, because those sets are contained in the
    # sets for prime divisors of p.
    #
    # In fact, to cover all b >= 2, we only need the union of perfect powers
    # with prime exponents.  Then we do inclusion-exclusion on those primes p1,p2,...
    # The intersection PP'(p1) ∩ PP'(p2) is all numbers that are a^(lcm(p1,p2)).
    # For distinct primes p1, p2, we have lcm(p1,p2) = p1*p2, etc.
    #
    # But a simpler way (and more intuitive to implement) is:
    #  1) Gather all products of distinct primes <= 60 (these products are
    #     the possible "lcm exponents").
    #  2) Use inclusion-exclusion with sign = (+1 if the subset has odd size,
    #     -1 if even).
    #  3) For a product e, the number of integers up to N that are a^e for a>1
    #     is floor(N^(1/e)) - 1 (subtracting the base=1 case).
    #  4) Sum these with alternating signs.
    #  5) Finally add +1 for x=1, which is 1^b for any b≥2.
    #
    # Implementation details:
    #  - We list all primes up to 59.
    #  - Generate all subsets of these primes (up to size 3, since any product
    #    of 4 or more distinct primes ≥ 2*3*5*7 = 210 > 60) whose product ≤ 60.
    #  - Compute floor(N^(1/product)) - 1 for each subset and accumulate with
    #    inclusion-exclusion.
    #  - Add 1 at the end to account for x=1.
    #
    # This method is efficient (only a few dozen subsets) and fits easily
    # within time and memory limits, even for N up to 10^18.
    # -------------------------------------------------------------------
    
    # List of primes up to 59
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    
    # We'll gather all subsets (with product ≤ 60) of these primes.
    # Because the product grows quickly, the largest subset size that
    # can still have product ≤ 60 is at most 3.  (2*3*5=30, 2*3*7=42, but
    # 2*3*5*7=210>60)
    subsets = []  # will hold tuples (product_of_primes_in_subset, subset_size)
    
    def backtrack(start_idx, curr_prod, count):
        # Record subsets as we go
        # Then try to include further primes
        for i in range(start_idx, len(primes)):
            p = primes[i]
            new_prod = curr_prod * p
            if new_prod > 60:
                break
            subsets.append((new_prod, count+1))
            backtrack(i+1, new_prod, count+1)
    
    backtrack(0, 1, 0)
    
    # We only need an integer k-th root function for k up to 60 (worst case),
    # which is not too large.  We'll use a simple binary search for floor(N^(1/k)).
    def kth_root(x, k):
        # Special quick cases
        if x < 2:
            return x
        low, high = 1, x
        while low <= high:
            mid = (low + high) // 2
            # mid^k comparison
            # Python's pow can handle large integers but we must be mindful of speed
            power = pow(mid, k)
            if power == x:
                return mid
            elif power < x:
                low = mid + 1
            else:
                high = mid - 1
        return high
    
    # Inclusion-exclusion over these subsets:
    total = 0
    for (exp_val, size_of_subset) in subsets:
        # how many a>1 satisfy a^exp_val <= N ?
        root_val = kth_root(N, exp_val)
        count_above_one = 0
        if root_val > 1:
            count_above_one = root_val - 1  # exclude a=1
        
        # Add or subtract based on parity of subset size
        if size_of_subset % 2 == 1:
            total += count_above_one
        else:
            total -= count_above_one
    
    # Finally, add +1 for x=1 (since 1=1^b for b>=2)
    answer = total + 1
    
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()