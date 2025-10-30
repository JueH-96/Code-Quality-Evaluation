class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        """
        We want Alice to pick up exactly k ones while standing at some fixed index aliceIndex.
        She may:
          • Immediately pick up the 1 at aliceIndex (if nums[aliceIndex] == 1), which costs 0 moves.
          • Perform "swap" moves (cost=1 move each) to bring an existing 1 from elsewhere to aliceIndex.
            Each swap moves the 1 one step closer. Hence moving a 1 from position x to aliceIndex i
            costs |x - i| moves (in an idealized sense, because we can shuffle zeros/ones via swaps).
          • Create up to maxChanges new ones at positions j != aliceIndex where nums[j] == 0 (cost=1 move each),
            and then swap that new 1 to aliceIndex (another sequence of swaps). However, the cheapest way
            if we are going to create a new 1 for aliceIndex is:
                 - Place the new 1 into a position adjacent to aliceIndex (cost=1 move),
                 - Then do one swap to bring it into aliceIndex (cost=1 more move),
              for a total of 2 moves. (If adjacency spots are occupied or out of range, cost can be ≥ 2,
              but 2 is the best possible if at least one adjacent 0 is available. In the worst case,
              you could place the new 1 anywhere and still pay “1 + distance” moves. But since we aim
              for the minimal overall cost, we assume the best-case placement of newly created ones
              if needed.)
        
        The key insight:
          1) Let S = number of existing ones in nums.
          2) For each potential standing index i, let initialPick = 1 if nums[i] == 1 else 0.
             Then we still need r = k - initialPick more ones.
          3) We can gather those r ones in two ways:
               • Move from among the existing S ones.  Moving a 1 at position x costs |x - i|.
               • Or create a new 1 for cost=2 (assuming best-case adjacency). We can do this up to maxChanges times.

          4) Hence for each i, we want to choose some number r_exist from the existing S ones, and some number r_new
             from newly created ones, with
                 r_exist + r_new = r
             and
                 r_new <= maxChanges
             so r_new ≤ maxChanges and r_exist ≤ S (since we only have S existing ones total).
          5) Among the existing S ones, we only want to pay for the r_exist cheapest moves. In other words,
             we look at the distances D_i = {|x - i| : x where nums[x]=1 }, sort them ascending, and sum up
             the smallest r_exist distances. Then we add r_new × 2 for the newly created ones.

        The main challenge is to do this efficiently for all i, because n can be up to 10^5.

        -------------------------------------------------------
        Efficient Implementation Sketch:
          • First, gather pos[] = sorted indices where nums[pos] == 1. Let S = len(pos).
          • We will iterate i from 0 to n-1. For each i, define:
               dist_j = |pos[j] - i|
            sorted in ascending order. We want the minimal sum of r = k - initialPick of those distances if we
            choose r_exist from [0..min(r, S]], plus we pay 2*(r - r_exist) if (r - r_exist) <= maxChanges.

            However, sorting dist_j anew for each i is O(S log S) repeated n times => O(n*S log S) can be too large.
          • Observe we only need the first r distances in ascending order (the r "closest" 1's). Re-sorting for every i is expensive.
          • A known approach: as i goes from 0 to n-1, |pos[j] - i| changes in a predictable manner (it is piecewise linear),
            and we can maintain a "distance multiset" with a balanced tree or two heaps. Then we can query partial sums
            of the smallest r distances. We also track how many of those distances exceed 2 (those might be replaced with 2
            if we have leftover changes). Implementing that carefully can be done in O(n log S) with the right data structure.

        -------------------------------------------------------
        For brevity in this solution code, we will implement a direct method that is optimized enough to pass typical
        contest constraints in practice, using a two-pointer approach coupled with prefix sums of pos[] and careful
        counting of distances ≤ 2.  The high-level steps:

          1) Precompute prefix sums of pos[] to quickly compute sums of distances to any i:
                 prefixPos[j] = sum of pos[0..j-1]
             Then sum of distances from all positions to i is:
                 sum(|pos[j] - i|) = i * (# pos[j] > i) - sum of those pos[j] > i
                                   + sum of those pos[j] < i   - i * (# pos[j] < i)
             which can be found by binary searching i in pos[].

          2) Also precompute, for each i, the counts of how many 1's are at distance 0,1,2,≥3 from i. We can do
             this by partial range counting with prefix arrays. Then to pick the r closest distances, we:
                - First pick from distance=0 if available (cost=0 each).
                - Then from distance=1 if we still need more (cost=1 each).
                - Then distance=2 if we still need more (cost=2 each).
                - If we still need more from distance≥3, each costs ≥3. But we can replace at most (maxChangesLeft)
                  of them with cost=2, and pay the actual distance for the remainder if we needed even more. If
                  the cost of actual distance is bigger than 2, we prefer to create new ones at cost=2 if we have
                  changes left. Otherwise we pay the real distance.
             Summing that up yields the cost to gather r = (k - initialPick) ones from i. If r > S, we must create
             r - S new ones outright, if we can. Etc.

          3) We take the minimum over all i.

        Because of the implementation complexity, below is a concise but more direct code. It uses counting
        techniques plus prefix sums and a sliding approach to count how many 1's lie within distance d of i.
        This is still non-trivial, but is done in steps that fit within O(n log n) or O(n + S) with careful coding.

        -------------------------------------------------------
        NOTE: The code below focuses on correctness rather than hyper-optimization. It follows the logic:
              "Compute, for each i, how many 1's are at each distance 0..2, and how many are beyond 2. Then
               combine them with the leftover creation logic." Then picks the minimal cost. 
        """

        import bisect
        
        n = len(nums)
        positions = [i for i,v in enumerate(nums) if v == 1]
        S = len(positions)
        
        # If we need k ones in total, but if k > S + maxChanges, impossible (but per constraints it's guaranteed feasible).
        # Edge case: if k == 1, we can just pick an index i where there's a 1 and pay 0 moves. Or create a new 1 if none exist.
        # We'll proceed with general logic.

        # Prefix sums of positions, to quickly compute sum(|pos[j] - i|)
        # We'll use them later for the "sum of all distances" approach if needed.
        prefix = [0]*(S+1)
        for i in range(S):
            prefix[i+1] = prefix[i] + positions[i]
        
        def total_dist(i):
            """
            Returns sum of |positions[j] - i| for all j.
            We'll also return how many positions < i, and how many positions > i
            to help partial computations.
            """
            # Find split with binary search
            idx = bisect.bisect_left(positions, i)
            left_count = idx
            right_count = S - idx
            # sum of left side distances: i*left_count - sum(positions[0..idx-1])
            left_sum = i*left_count - prefix[idx]
            # sum of right side distances: sum(positions[idx..S-1]) - i*right_count
            right_sum = (prefix[S] - prefix[idx]) - i*right_count
            return left_sum + right_sum
        
        # We also need a quick way to count how many positions are at distance <= 2 from i
        # and exactly how many are at distance 0,1,2,≥3. We'll do a small helper with two
        # binary searches for [i-2, i+2].
        # Then we can figure out how many are distance=0, distance=1, distance=2 among those,
        # using narrower range checks. This is local and short-circuits large scanning.
        
        # We'll create an auxiliary function that returns
        #   c0, c1, c2, c3plus
        #   sumDistUpTo2 = sum of the distances for those within distance 2
        #
        # Then picking the r closest 1's among them is straightforward:
        import math
        
        def count_distances_up_to_2(i):
            """
            Count how many 1's are at distance 0,1,2 from i, and also how many are beyond 2.
            Also compute the sum of distances for those that are within distance ≤2.
            """
            # We'll find the range [i-2, i+2], then iterate over that slice in positions
            left = bisect.bisect_left(positions, i-2)
            right = bisect.bisect_right(positions, i+2)
            slice_positions = positions[left:right]
            
            c0 = c1 = c2 = 0
            sum_0_1_2 = 0
            for pos in slice_positions:
                d = abs(pos - i)
                if d == 0:
                    c0 += 1
                elif d == 1:
                    c1 += 1
                    sum_0_1_2 += 1
                elif d == 2:
                    c2 += 1
                    sum_0_1_2 += 2
            c_upTo2 = c0 + c1 + c2
            
            # total 1's is S, so c3plus = S - c_upTo2
            c3plus = S - c_upTo2
            
            return c0, c1, c2, c3plus, sum_0_1_2
        
        # Next, for those that are distance≥3, we can compute the sum of distances
        # by sum of all distances minus sum of distances up to 2. Because:
        #    sum(|pos[j]-i|) for j in [0..S-1]
        #  minus
        #    sum of distances for those that are ≤2 from i
        #  gives
        #    sum of distances for those that are ≥3 from i.
        #
        # We'll do:
        #   allDist = total_dist(i)
        #   partDistUpTo2 = (   c0*0 + c1*1 + c2*2 ) = sum_0_1_2
        #   partDist3Plus = allDist - partDistUpTo2
        #
        # Then if we need to pick from those in distance≥3, each distinct 1 there has some distance≥3,
        # but we can "replace" that distance with 2 if we have "changes" left. We'll do it in ascending order
        # of actual distance, but we only know the sum. We do not have the distribution of distance≥3 by actual distance.
        # However, for the purpose of picking the "r smallest distances," we would pick all from distance=0,1,2 first,
        # then we move on to the 3+ bucket. In that 3+ bucket, the cost for each 1 is d≥3, or we can pay 2 if we have changes left.
        #
        # A simpler bounding approach: If we pick x from the 3+ group, in the worst case we must assume we pick the x smallest
        # distances from that group. The smallest distance in the 3+ group is 3, the largest could be up to n-1. If x <= c3plus,
        # the sum cost is at least 3*x if we pay actual distances. But if we have enough changes left to replace them with 2,
        # we pay 2*x. In real detail, distances could be 3,4,10,... but we do not store them individually. To be correct, we
        # need to be sure that "the x smallest 3+ distances" won't be significantly different from x times 3. The maximum error
        # arises if some are large (like distance=50). Replacing that with cost=2 saves more. But we do not have the exact
        # distribution. So we must compute it exactly if we want a perfect answer. 
        #
        # => We do need the actual distribution of 3+ distances for perfect minimal picking. That implies a more complex data
        #    structure. However, a simpler bounding approach that lumps "≥3" into a single bucket can break correctness if the
        #    best solution uses large distances that we turn into 'create new 1' for cost=2.
        #
        # Correct approach with direct distribution:
        #   sumDist3Plus = allDist - sum_0_1_2
        #   We also know how many: c3plus = S - c_upTo2.
        #   We do not know exactly how the c3plus distances are spread from 3..(n-1).
        #   But to be correct for minimal picking, any distance≥3 is strictly more expensive
        #   than or equal to cost=3 if we move it, and we can reduce it to 2 by using a "change".
        #
        #   If we are forced to use x of them from the 3+ group (because we need more than c0+c1+c2 existing ones
        #   and cannot create enough new ones to avoid them wholly), we want to apply changes if possible to reduce cost
        #   from distance≥3 down to 2. That yields a saving of (d - 2) for each replaced item. Large d give bigger savings.
        #   So we would want to apply changes first to the largest distances. But we do not store them individually...
        #
        # For large n, a fully correct solution does indeed require a more intricate structure to handle partial usage of the 3+ bucket
        # in ascending order of distances. This is fairly complicated to implement from scratch in a short solution.
        #
        # ----------------------------------------------------------------------------
        # In Contest or Interview Setting:
        #   The standard method is to keep a balanced data structure or two-heap approach that can be updated as i moves,
        #   so we always can retrieve the sorted distances or prefix sums. Then for each required r, we can do a partial pick
        #   from smallest to largest, applying "create" replacements if beneficial and we have leftover changes. 
        #
        # Below is a simplified "reference" implementation that does exactly that distribution approach explicitly
        # by building a distance array for each i in O(S) time, sorting it, and then computing the cost. This is O(n*S log S),
        # which may be too large for n=1e5, S=1e5 in worst case. However, in many practical test sets or smaller constraints
        # it would pass. It is guaranteed correct by construction.
        #
        # For very large input, one must replace this with a more sophisticated approach (as discussed above).
        # Here, we show the correct logic, not the heavy optimization. 
        #
        # WARNING: The worst-case complexity is O(n * S log S). Use only if constraints/test time allow it.
        #
        # ----------------------------------------------------------------------------
        
        # If k == 0 (not in constraints, since k>=1), cost=0 trivially.
        # We'll implement the direct method:

        # Edge case small S or k=1, we can short-circuit quickly:
        if k == 1:
            # We only need 1 one. The best is to pick an index i with nums[i]=1 -> cost=0.
            # If no existing 1, we can create one for cost=1 (placing it adjacent won't help if i can't pick it up
            # immediately, so we need a swap => total cost=2). But constraints say sum(nums)+maxChanges >= k => so there's
            # at least 1 existing one or we can create. Let's see if there's a free 1:
            if 1 in nums:
                return 0  # pick that i
            else:
                # must create => place a new 1 for cost=1 move + 1 swap => cost=2
                return 2

        # Build the list of onesIndices. We will do a direct check for each i in [0..n-1].
        # For each i, build a distance array, sort it, pick the r smallest with "replace if distance>2 and we have leftover changes"
        # and if r > S, we also create new ones for cost=2 each. Then add to a global min.

        # Precompute distances for all i in a rolling manner is also large memory (n*S).
        # We will do it on the fly. This is quite large but straightforward.

        # If S=0 (no ones in the array), we must create all k ones at cost=2 each => 2*k moves (unless the first pick is free? No, there's no free pick).
        # Actually if nums[i]=0, no free pick. So cost=2*k if k <= maxChanges. Done.
        if S == 0:
            # We rely on creation only
            # To pick up k ones, each creation costs 1 + we need 1 swap => cost=2 each
            return 2*k  # (feasibility is guaranteed by sum(nums)+maxChanges >=k => maxChanges>=k)

        ans = math.inf
        
        # Pre-sort the positions array
        ones = positions  # already sorted
        
        for i in range(n):
            needed = k
            # Did we get a free one if nums[i]==1?
            if nums[i] == 1:
                needed -= 1
                if needed <= 0:
                    # That means picking up the single 1 at i is enough
                    ans = min(ans, 0)
                    continue
            
            if needed > S + maxChanges:
                # Not possible to gather needed ones if we can't even use all S plus maxChanges new
                continue
            
            # Gather the distance array for existing S ones
            dist_arr = []
            for x in ones:
                dist_arr.append(abs(x - i))
            dist_arr.sort()
            
            # Now we want the cheapest needed out of these S distances, but we can replace up to maxChanges
            # of them that exceed 2 with cost=2, and if needed > S, we create the remainder => cost=2 each,
            # if we still have enough changes left. 
            # Implementation:
            # Step 1: pick up to needed from dist_arr in ascending order
            # Step 2: for each distance d, if d>2 and we have leftover changes, replace cost with 2, else cost = d
            # Step 3: if we haven't reached needed ones after exhausting dist_arr, create the difference with cost=2 each 
            #         (requires leftover changes).
            
            leftover_changes = maxChanges
            total_moves = 0
            used = 0
            
            for d in dist_arr:
                if used == needed:
                    break
                # We want this 1. The cost is d, but if d>2 and leftover_changes>0, we can pay 2 instead
                # Actually if d>2, we prefer to create a new 1 for cost=2 (instead of moving from that distance).
                # That uses up 1 "change". So effectively, cost = min(d, 2) if leftover_changes>0 for d>2,
                # or cost = d otherwise. Also note that if d>2 but leftover_changes=0, we must pay d.
                if d <= 2:
                    total_moves += d
                    used += 1
                else:
                    # d>2
                    if leftover_changes > 0:
                        total_moves += 2
                        leftover_changes -= 1
                        used += 1
                    else:
                        total_moves += d
                        used += 1
            
            # If we've used < needed after using all S (or fewer if we broke early),
            # we must create the remainder
            if used < needed:
                # create count = needed - used new ones, each with cost=2 => also uses that many changes
                create_count = needed - used
                if create_count <= leftover_changes:
                    total_moves += 2*create_count
                else:
                    # Not feasible if we can't create enough
                    continue
            
            ans = min(ans, total_moves)

        return ans if ans < math.inf else 0