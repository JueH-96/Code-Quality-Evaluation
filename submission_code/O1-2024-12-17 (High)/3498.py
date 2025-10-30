class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        """
        We want to make (possibly zero) replacements of elements in 'nums' (each element can be changed
        to any integer in [0..k]) so that there is a single integer X with:
            abs(a[i] - a[n - 1 - i]) = X  for all i

        We need to minimize the number of changes.

        Key idea:
        Let n be even. We have n/2 "pairs": (nums[i], nums[n-1-i]).  
        For one pair (a, b), define d = abs(a-b).

        After all changes, every pair must have the same absolute difference X. For each pair:
          - Cost 0 if it already has difference d == X (no change needed).
          - Cost 1 if we can change exactly one element of the pair to achieve difference X.
          - Cost 2 otherwise (changing both elements can achieve any difference from 0..k).

        Observing constraints:
          0 <= a,b <= k
          If we change one element of (a, b) to a' in [0..k], we can achieve X iff a' = b+X or b-X
          is within [0..k], which is X <= b or X <= k-b.  Similarly for changing b. 
          Hence, for one change, we can achieve X in [0..max(a, k-a)] or [0..max(b, k-b)].
          Overall for one pair, feasible X with cost=1 is any X <= max( max(a,k-a), max(b,k-b) ). 
          (Call that value r.)

        So the per-pair cost function C_i(X) is:
          C_i(X) = 0 if X == d
                   1 if 0 <= X <= r and X != d
                   2 if X > r

        We combine pairs by summing costs:  Sum_i C_i(X).  We want to pick X in [0..k] minimizing that sum.

        To do this efficiently, we use a "difference array" approach (sometimes called a "sweep-line"):

          For each pair (a,b):
            let d = abs(a-b), r = max( max(a,k-a), max(b,k-b) ).

            1) Start cost=2 for X in [0..k] => in a difference array diff, do:
                  diff[0]   += 2
                  diff[k+1] -= 2
            2) Subtract 1 for X in [0..r], turning cost from 2 down to 1 => 
                  diff[0]   -= 1
                  diff[r+1] += 1
            3) If d <= r, subtract 1 at X=d only, making cost=0 => 
                  diff[d]   -= 1
                  diff[d+1] += 1

        Summing these updates for all pairs yields a final difference array.  Then we do a prefix sum
        over [0..k] to get total cost at each X.  The minimum of those values is our answer.

        Complexity: O(n + k).

        Example walk-through matches the approach and yields the correct results.
        """

        n = len(nums)
        half = n // 2
        
        # We'll maintain a difference array of length k+2 so we can safely do
        # index+1 updates at "k+1" without out-of-bounds.
        diff = [0] * (k + 2)
        
        def M(x):
            # M(x) = max(x, k-x), helps compute the one-change feasibility range
            return max(x, k - x)
        
        # Accumulate the difference-array updates for each pair
        for i in range(half):
            a, b = nums[i], nums[n - 1 - i]
            d = abs(a - b)
            r = max(M(a), M(b))  # largest feasible difference with one change
            
            # Base: cost = 2 for all X
            diff[0]     += 2
            diff[k + 1] -= 2
            
            # Reduce (2->1) over range [0..r]
            diff[0]     -= 1
            diff[r + 1] += 1
            
            # If d <= r, reduce cost from 1 to 0 at X = d
            if d <= r:
                diff[d]     -= 1
                diff[d + 1] += 1
        
        # Now build the actual cost array by prefix summation and find the minimum
        current_cost = 0
        answer = float('inf')
        for x in range(k + 1):
            current_cost += diff[x]
            if current_cost < answer:
                answer = current_cost
        
        return answer