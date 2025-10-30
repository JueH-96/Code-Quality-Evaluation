class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        # We'll use a digit DP method to count all beautiful numbers in [0, N].
        # A beautiful number is one for which the product of its digits is divisible by the sum of its digits.
        # We define a helper function count_beautiful(N) that returns the count of beautiful numbers in [0, N].
        # Later, the answer for [l, r] will be count_beautiful(r) - count_beautiful(l-1),
        # taking care that l is positive.
        
        def count_beautiful(N: int) -> int:
            sN = str(N)
            n = len(sN)
            # Use recursion with memoization.
            # State parameters:
            # pos: current position in the number (0-indexed)
            # tight: whether the prefix is exactly equal to the prefix of N (True/False)
            # started: whether we have already placed a non-zero digit (True/False)
            # summ: the sum of digits chosen so far
            # prod: the product of digits chosen so far.
            #
            # Note: When we haven't started (leading zeros), we keep summ = 0 and prod = 1.
            # When we eventually add a non-zero digit, we update accordingly.
            #
            # When we finish (pos == n), if not started, then the number is 0 (not positive) and we return 0.
            # Otherwise, if summ > 0, and the product is divisible by the sum, we count 1.
            
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dfs(pos, tight, started, summ, prod):
                if pos == n:
                    if not started:
                        return 0  # number 0 is not a positive integer
                    # Check the beautiful condition: product is divisible by sum.
                    if summ > 0 and prod % summ == 0:
                        return 1
                    else:
                        return 0
                limit = int(sN[pos]) if tight else 9
                total = 0
                for d in range(limit+1):
                    new_tight = tight and (d == limit)
                    if not started and d == 0:
                        # still haven't started the number; sum and product remain unchanged.
                        total += dfs(pos+1, new_tight, False, summ, prod)
                    else:
                        # Once we start the number, update the sum and product.
                        # If we are starting now (i.e. not started before and d > 0), we set product = d.
                        # Otherwise, if already started, product *= d.
                        new_prod = (prod * d) if started else d
                        total += dfs(pos+1, new_tight, True, summ + d, new_prod)
                return total
            
            return dfs(0, True, False, 0, 1)
        
        # Compute the count in [l, r].
        # For l = 1, count_beautiful(l-1) would count numbers in [0,0], which is 0.
        return count_beautiful(r) - count_beautiful(l-1) if l > 1 else count_beautiful(r)


# Below are some tests:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    l, r = 10, 20
    print(sol.beautifulNumbers(l, r))  # Expected output: 2 (Beautiful numbers: 10 and 20)
    
    # Example 2:
    l, r = 1, 15
    print(sol.beautifulNumbers(l, r))  # Expected output: 10 (Beautiful numbers: 1,2,3,4,5,6,7,8,9,10)