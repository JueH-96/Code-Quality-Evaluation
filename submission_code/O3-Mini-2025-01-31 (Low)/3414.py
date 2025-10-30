class Solution:
    def waysToReachStair(self, k: int) -> int:
        # The key observation is that any sequence of operations is composed of two types:
        # “up‐operations” that increase the stair number and “down‐operations” that decrease it.
        # An up‐operation, when performed with the current jump index j, moves from stair i to i + 2^j,
        # and increases jump by 1.
        # A down‐operation simply moves one stair down (i.e. from i to i – 1), but it is not allowed
        # to have two down‐operations consecutively (and cannot be used on stair 0).
        #
        # It turns out one may “compress” the possible sequences by observing that the up-operations
        # always occur in order (they must update the jump counter consecutively) and they give a fixed total
        # upward displacement. Down-operations, however, can be inserted in any of the "gaps"
        # between the up-moves (including before the first up and after the last up), with the constraint
        # that there can be at most one down in each gap (because consecutive downs are forbidden).
        #
        # Suppose we perform x up-operations. Then:
        #  * The jth up operation (with j=0,...,x-1) adds 2^j.
        #  * Total upward displacement = 2^0 + 2^1 + ... + 2^(x-1) = 2^x - 1.
        # We begin at stair 1, so after the ups the position is:
        #    1 + (2^x - 1) = 2^x.
        #
        # Next, suppose we choose to insert exactly y down-operations.
        # Down operations reduce the position by y so the final position is:
        #    pos = 2^x - y.
        #
        # In order for pos == k we need:
        #    2^x - y = k,   i.e., y = 2^x - k.
        #
        # However, we are allowed to insert at most one down-operation per "gap".
        # These gaps occur:
        #    - before the first up
        #    - between consecutive ups (there are (x - 1) such gaps)
        #    - after the last up
        # Total gaps = x + 1.
        # Thus we must have:
        #    0 <= y <= x + 1.
        #
        # Also, note x can be 0 (i.e. no up-operations). In that case:
        #    Position = 1 - y.
        # For pos to equal k, we need: 1 - y = k, so y = 1 - k.
        # This is possible only if k <= 1.
        #
        # In summary, for each possible x (number of ups) the sequence leads to stair k if and only if:
        #    y = 2^x - k is an integer with 0 <= 2^x - k <= x + 1.
        # And if so, there are C(x+1, y) ways to choose which gaps contain the down-operation.
        #
        # The answer is therefore the sum over all valid x of combinations: 
        #     Sum_{x such that 0 <= 2^x - k <= x+1} C( x+1, 2^x - k ).
        #
        # Since k can be as large as 10^9, note that 2^x grows exponentially and we need only consider x up to around 31-32.
        # We will simply loop x from 0 up to 60 (a safe upper bound) and sum valid contributions.
        
        # Precompute combination (binomials) for needed values.
        # Since x+1 is at most around 60, we can compute C(n, k) using math.comb (Python 3.8+)
        from math import comb
        
        total = 0
        # x is the number of up operations.
        # Let x_max be a number such that 2^x > k is guaranteed to exceed k for sure.
        # We choose a small upper limit because the possibilities are very limited.
        for x in range(61):
            # Compute the total upward displacement if x up operations are performed.
            up_total = (1 << x)  # 2^x
            # The required number of down operations is:
            y = up_total - k
            if y < 0 or y > (x + 1):
                continue
            # Add ways: choosing y positions out of x+1 gaps (each gap can at most
            # contain one down-operation).
            total += comb(x + 1, y)
        return total


# Sample test runs:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.waysToReachStair(0))  # Expected output: 2
    # Example 2:
    print(sol.waysToReachStair(1))  # Expected output: 4