from math import ceil
from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # If there is only one element, no operations are needed.
        if n == 1:
            return 0
            
        A = sum(nums)
        mx = max(nums)
        mn = min(nums)
        
        # It might be never beneficial to pair if two singles cost no more.
        # (i.e. if  cost2 >= 2*cost1, then doing two single increments is at most as expensive.)
        if cost2 >= 2 * cost1:
            total_increments = n * mx - A  # each number increased to mx
            return (total_increments % MOD) * cost1 % MOD
        
        # When there are two numbers, note that if one is already at mx then pairing never
        # can be done, because pairing requires two different indices to have pending increments.
        if n == 2:
            # Let a <= b. To equalize, we must bring a up to b.
            return ((mx - mn) % MOD) * cost1 % MOD

        # For n >= 3 and when pairing is beneficial (cost2 < 2*cost1),
        # we can choose a target value t (>= max(nums)) so that when every element is increased
        # to t, the cost is minimized.
        #
        # For each index, d[i] = t - nums[i]. We have S = sum(d[i]) = n*t - A.
        # We can combine increments for two different indices in one operation at cost cost2.
        # However, we can only pair an increment from one index with another if both still need increments.
        # A known fact in such “pairing” problems is that the maximum number P of pair operations
        # you can do is
        #      P = min( S//2, S - max(d[i]) ).
        # Since max(d[i]) = t - mn, we may write:
        #      P = min( S//2, (n*t - A) - (t - mn) )
        #        = min( S//2, (n-1)*t - (A - mn) ).
        #
        # After using P pair operations (covering 2*P increments at cost cost2 each),
        # the remaining S - 2P increments must be done individually at cost cost1.
        # So the total cost function is:
        #
        #      f(t) = P * cost2 + (S - 2*P)*cost1,
        #             where S = n*t - A and P = min(S//2, (n-1)*t - (A - mn)).
        #
        # Our goal is to choose an integer t (with t >= mx) to minimize f(t).
        #
        # A useful observation is that if we “overshoot” t (choose t > mx) we add extra increments
        # to all numbers. However, a more balanced profile may allow us to pair more increments
        # (since the index with the minimum value now has an even larger gap). In some cases – for instance,
        # when one number is very small compared to the others – it might be best to overshoot.
        #
        # We can search for the best integer t in a range.
        # A natural lower bound is t = mx.
        # For the upper bound, note that the possibility of pairing is limited by the “imbalance”
        # between the smallest and the other entries; indeed, max(d[i]) = t - mn.
        # To have a chance to “balance” out the increments, one sensible candidate is when
        #    (n-1)*t - (A - mn)  becomes comparable to  S//2.
        # Ignoring floors, setting S/2 = (n-1)*t - (A - mn)
        #   => (n*t - A)/2 = (n-1)*t - (A - mn)
        #   => Multiply both sides by 2: n*t - A = 2(n-1)t - 2A + 2mn
        #   => Rearranging: (n-2)t = A - 2mn
        #   => t = (A - 2*mn) / (n-2).
        #
        # Thus, we choose an upper bound for t as:
        #     high_bound = max(mx, ceil((A - 2*mn) / (n-2)))
        # (If A - 2*mn <= 0 then the array is already equal and the answer is 0.)
        
        # Helper function to compute total cost for a given target t.
        def f(t: int) -> int:
            S = n * t - A  # total increments needed
            # R_val = (n-1)*t - (A - mn) is the total increments available from the "other" numbers
            R_val = (n - 1) * t - (A - mn)
            # Maximum pair operations we can perform:
            P = min(S // 2, R_val)
            return P * cost2 + (S - 2 * P) * cost1
        
        lowCandidate = mx
        if A - 2 * mn > 0:
            highCandidate = ceil((A - 2 * mn) / (n - 2))
            if highCandidate < lowCandidate:
                highCandidate = lowCandidate
        else:
            highCandidate = lowCandidate
        
        # In many such problems the cost function f(t) (defined on integers t >= mx)
        # is unimodal. We can perform a binary search (or discrete ternary search)
        # to find the t producing the minimum cost.
        lo, hi = lowCandidate, highCandidate
        while lo < hi:
            mid = (lo + hi) // 2
            if f(mid) <= f(mid + 1):
                hi = mid
            else:
                lo = mid + 1
        ans = f(lo) % MOD
        return ans % MOD


# ----------------------------
# For testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minCostToEqualizeArray([4,1], 5, 2))  # expected output: 15
    # Example 2:
    print(sol.minCostToEqualizeArray([2,3,3,3,5], 2, 1))  # expected output: 6
    # Example 3:
    print(sol.minCostToEqualizeArray([3,5,3], 1, 3))  # expected output: 4

    # Additional tests:
    # When pairing is very beneficial (cost2 < 2*cost1), the function may choose to overshoot mx.
    # For instance, with nums = [3,5,3], cost1 = 1, cost2 = 1:
    #   If we use t = 5 (the natural candidate) the differences are [2,0,2].
    #   We can pair the two 2’s fully: 2 pair operations, cost = 2 * 1 = 2.
    #   So we expect output 2.
    print(sol.minCostToEqualizeArray([3,5,3], 1, 1))  # expected output: 2

    # When there are two elements the optimum is always to bring the smaller one up.
    # For nums = [1,100], cost1 = 100, cost2 = 1:
    #   Only one element needs increments: 100 - 1 = 99 increments, at cost 100 each.
    #   Expected cost = 9900.
    print(sol.minCostToEqualizeArray([1,100], 100, 1))  # expected output: 9900

    # A case with three numbers where overshooting is beneficial.
    # For nums = [1,2,100], cost1 = 100, cost2 = 1.
    #   Using target t = 100 gives differences [99,98,0] and you cannot pair much.
    #   Slightly increasing t lets even the largest element get extra increments so it can be paired.
    #   The optimal total cost is determined by our search.
    print(sol.minCostToEqualizeArray([1,2,100], 100, 1))