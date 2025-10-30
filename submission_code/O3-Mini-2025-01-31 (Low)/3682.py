MOD = 10**9 + 7

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        # A "good" array of length n with exactly k adjacent equal pairs
        # implies it has exactly s = n - k segments (each segment comprising equal numbers).
        # For example, if n = 3 and k = 1, then s = 2 segments.
        #
        # How to count these arrays?
        # 1. Choose positions for segment boundaries. There are (n - 1) gaps between numbers.
        #    Out of these, exactly (s - 1) positions must be chosen as boundaries where the value changes.
        #    So, there are C(n - 1, s - 1) choices.
        #
        # 2. For the first segment, we can choose any number from 1 to m (m choices).
        #    For each subsequent segment, we must choose a number different from the previous segment,
        #    resulting in (m - 1) choices per segment.
        #    Since there are (s - 1) subsequent segments, that gives (m - 1)^(s - 1) choices.
        #
        # Multiplying these together, the answer is:
        #    answer = C(n - 1, s - 1) * m * (m - 1)^(s - 1) mod MOD,
        # where s = n - k.
        #
        # Precompute factorials to efficiently compute combinations.
        
        segments = n - k  # number of segments
        if segments <= 0 or segments > n:
            return 0
        
        # Precompute factorials and inverse factorials up to n for combinations
        max_val = n  # since we need to compute up to (n-1)! 
        fact = [1] * (max_val + 1)
        invfact = [1] * (max_val + 1)
        for i in range(2, max_val + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        # Compute modular inverse using Fermat's Little Theorem
        invfact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        for i in range(max_val, 0, -1):
            invfact[i - 1] = (invfact[i] * i) % MOD
        
        # Compute the combination: C(n - 1, segments - 1)
        comb_val = fact[n - 1] * invfact[segments - 1] % MOD * invfact[(n - 1) - (segments - 1)] % MOD
        
        # Calculate the answer
        ways = comb_val * m % MOD * pow(m - 1, segments - 1, MOD) % MOD
        return ways

# Below are a few tests to validate the solution:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    # Input: n = 3, m = 2, k = 1
    # Expected Output: 4
    print(sol.countGoodArrays(3, 2, 1))  # Output: 4
    
    # Example 2:
    # Input: n = 4, m = 2, k = 2
    # Expected Output: 6
    print(sol.countGoodArrays(4, 2, 2))  # Output: 6
    
    # Example 3:
    # Input: n = 5, m = 2, k = 0
    # Expected Output: 2
    print(sol.countGoodArrays(5, 2, 0))  # Output: 2