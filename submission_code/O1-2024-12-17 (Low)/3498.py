class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        """
        We have an even-length array nums and an integer k. We want to make
        some (possibly zero) replacements of elements (each replaced element
        can become any integer in [0..k]) so that, in the final array, there
        is a single integer X satisfying:

            abs(nums[i] - nums[n - 1 - i]) = X   for all i in [0..(n/2)-1].

        We need to find the minimum number of changes required.

        --------------------------------------------------------------------
        APPROACH (Difference Array / "Sweep Line" for costs):
        
        Let n be even. We pair up indices: (i, n-1-i), for i = 0..(n/2 - 1).
        For each pair (L, R) = (nums[i], nums[n-1-i]), L,R in [0..k], define:
          d0 = |L - R|  (the current difference with 0 changes)
        
        If we pick a target difference X, the cost (number of elements to change)
        for that pair is:
            cost(L,R,X) = 0   if |L - R| == X
                         = 1   if we can fix difference to X with exactly 1 change
                         = 2   otherwise
                         
        - cost = 0 means no changes needed => this happens exactly when d0 == X.
        - cost = 1 means we can keep either L or R and change the other so that
          |L' - R| = X or |L - R'| = X with the new value in [0..k].
          That is possible if (L ± X) is in [0..k] or (R ± X) is in [0..k].
          (Because to force the difference to X by changing one side, we must set
           R' = L ± X or L' = R ± X, so long as the new value stays in [0..k].)
        - cost = 2 if neither of the above cases holds.

        We then want to pick X in [0..k] that minimizes sum of cost(L_i, R_i, X)
        over all i.

        A direct per-(pair, X) calculation for large k might be too big
        (O(n * k) could be up to 1e10). Instead, we use a difference-array trick
        often used for "range increment" problems:

        Step-by-step:
         1. Initialize a diff array of length k+2 with all zeros.  We'll build
            an eventual "cost array" = prefix-sum of diff over [0..k].
            Intuitively, we'll start by assuming cost=2 for each pair for all X,
            then decrement the cost in ranges/points where fewer changes suffice.

         2. For each pair (L, R):
            Let M_L = max(L, k-L).   (the largest X that can be made by changing R alone)
            Let M_R = max(R, k-R).   (the largest X that can be made by changing L alone)
            M = min(k, max(M_L, M_R))  # clamp to k

            a) Initially add +2 to the entire range [0..k], meaning "cost=2"
               for this pair. We do:
                  diff[0]   += 2
                  diff[k+1] -= 2
               
            b) For X in [0..M], cost can be at most 1 (changing exactly one element).
               So we subtract 1 over the range [0..M]:
                  diff[0]   -= 1
                  diff[M+1] += 1      (if M < k, to end that decrement)
            
            c) If d0 <= k:
               - If d0 <= M, then in that region we had cost=1 after step (b),
                 but actually cost=0 at X=d0, so subtract an additional 1 there.
               - If d0 > M, then cost was still 2 at X=d0, so we subtract 2
                 to make it 0 at that single point.
               
               In either case, we do:
                 if d0 <= M:
                    diff[d0]   -= 1
                    diff[d0+1] += 1
                 else:
                    diff[d0]   -= 2
                    diff[d0+1] += 2

         3. After processing all n/2 pairs, we take the prefix sums of diff to get
            the final cost array costX[0..k]. The answer is min(costX[x] for x in [0..k]).

        This yields an O(n + k) solution, which is feasible for n,k up to 1e5.

        We test with the given examples to validate.

        --------------------------------------------------------------------
        """

        import sys
        input_data = nums  # just naming consistency

        n = len(input_data)
        half = n // 2

        # We'll need a difference array of length k+2 to handle border updates safely
        diff = [0]*(k+2)

        # Process each pair
        left = 0
        right = n - 1
        for _ in range(half):
            L = input_data[left]
            R = input_data[right]
            left += 1
            right -= 1

            d0 = abs(L - R)
            # M = maximum X for which cost can be 1 (by changing exactly one of L,R)
            # We can keep L if we can place R' in [0..k], which covers X up to max(L, k-L).
            # Similarly keep R covers X up to max(R, k-R).
            # We combine these and clamp to k.
            M_L = max(L, k - L)
            M_R = max(R, k - R)
            M = min(k, max(M_L, M_R))

            # (a) Assume cost=2 for all X => diff[0]+=2, diff[k+1]-=2
            diff[0] += 2
            diff[k+1] -= 2

            # (b) For X in [0..M], cost at most 1 => subtract 1 over [0..M]
            diff[0] -= 1
            if M+1 <= k:
                diff[M+1] += 1

            # (c) For X=d0 (if d0 <= k), cost=0 => subtract the difference vs. the
            #     cost in that region:
            #     - if d0 <= M, that region's cost is currently 1 => subtract 1 more
            #     - if d0 > M, that region's cost is currently 2 => subtract 2 more
            if d0 <= k:
                if d0 <= M:
                    diff[d0] -= 1
                    diff[d0+1] += 1
                else:
                    diff[d0] -= 2
                    diff[d0+1] += 2

        # Now compute the prefix sums to get the actual cost array
        curr = 0
        ans = float('inf')
        for x in range(k+1):
            curr += diff[x]
            ans = min(ans, curr)

        return ans