class Solution:
    def minOperations(self, k: int) -> int:
        # If k is 1, then the initial array [1] already meets the condition.
        if k == 1:
            return 0
        
        # The most efficient strategy is to:
        # 1. Pick a target value x.
        # 2. Increase the initial element from 1 to x (cost = x - 1 increments).
        # 3. Duplicate that element enough times so that (number of copies) * x >= k.
        #
        # Suppose we perform p duplications; then we have (p + 1) copies of x.
        # We require (p + 1) * x >= k, so p must be at least ceil(k / x) - 1.
        # Total operations = (x - 1) + (ceil(k / x) - 1) = x + ceil(k / x) - 2.
        #
        # Our goal is to minimize: f(x) = x + ceil(k / x) - 2.
        
        ans = float('inf')
        # Try all possible x values from 1 to k
        for x in range(1, k + 1):
            # Compute the duplications needed: ceil(k / x) - 1.
            # Using integer arithmetic: ceil(k / x) can be computed as (k + x - 1) // x
            duplications = (k + x - 1) // x - 1
            cost = (x - 1) + duplications  # Total operations: increments + duplications
            ans = min(ans, cost)
        return ans

# You can run a simple test case as follows:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minOperations(11))  # Expected output: 5
    print(sol.minOperations(1))   # Expected output: 0