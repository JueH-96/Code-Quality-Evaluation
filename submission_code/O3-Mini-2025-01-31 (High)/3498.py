from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        half = n // 2
        # Worst‐case: every pair needs 2 changes.
        base = 2 * half

        # For each pair, if target X is “easy” to obtain (with one change)
        # we can subtract 1 from the cost.
        # And if the current pair already naturally yields X (i.e. |a-b|==X) then
        # we can get cost 0 for that pair.
        #
        # Our plan is to “simulate” for every possible X in [0,k] the total cost.
        # Instead of iterating over all pairs for each X explicitly (which would be too slow),
        # we use a difference‐array technique.
        #
        # For a given pair (a, b):
        #   • In the worst case the cost is 2.
        #   • If X can be made with one change then the cost reduces to 1.
        #   • And if X equals the natural difference (|a-b|) then cost is 0.
        #
        # Thus we “start” with cost 2 for every pair and for each pair subtract 1 for all X
        # that are “reachable” in one move and subtract an extra 1 at X = |a-b|.
        
        # diff will be used to mark that for this pair, for all X <= one_move_cap the cost decreases by 1.
        # We make diff of length (k+2) so that we can mark diff[one_move_cap+1] for the endpoint.
        diff = [0] * (k + 2)
        # count_d marks, for a given target X, how many pairs already yield difference X (cost=0 instead of 1)
        count_d = [0] * (k + 1)
        
        for i in range(half):
            a = nums[i]
            b = nums[n - i - 1]
            d = abs(a - b)
            count_d[d] += 1
            # With 0 moves the pair gives difference d.
            # With one move, by changing one element, we can choose any value in [0, k].
            # In fact, if we keep one element fixed (say b), then we want a new value x such that:
            #      |x - b| = X.
            # This can be achieved provided either (b+X) <= k or (b-X) >= 0.
            # Hence one move is possible if X <= max(b, k-b).
            # Similarly by fixing a.
            # Thus for this pair, any target X in range [0, one_move_cap] is reachable with one move.
            one_move_cap = max(max(a, k - a), max(b, k - b))
            # For every X in [0, one_move_cap] we get a saving of 1 change.
            diff[0] -= 1
            diff[one_move_cap + 1] += 1

        # Now, for a given target X the cost for a pair is:
        #   0    if X == |a-b|
        #   1    if X is within the one-move achievable range (and X != |a-b|)
        #   2    otherwise.
        # Summing over all pairs, starting with base cost 2 per pair,
        # we subtract 1 for every pair which can get X with one move,
        # and subtract another 1 for every pair that already is good (X == |a-b|).
        #
        # So total_cost(X) = base + (sum of diff adjustments at X) - count_d[X]
        best = float('inf')
        currentAdjustment = 0
        for X in range(0, k + 1):
            currentAdjustment += diff[X]
            cost = base + currentAdjustment - count_d[X]
            best = min(best, cost)

        return best

# For quick local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minChanges([1,0,1,2,4,3], 4))  # Expected output: 2
    # Example 2:
    print(sol.minChanges([0,1,2,3,3,6,5,4], 6))  # Expected output: 2