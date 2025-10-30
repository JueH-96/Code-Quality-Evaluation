class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        """
        We want the minimum number of moves (each move is either creating a 1 in a zero-position
        or swapping an adjacent (1,0)) so that Alice can pick up exactly k ones.

        Key insight:
        ----------------------------------------------------------------------------
        1) Ignoring creation of new 1s, the cost to "collect" k of the existing 1s
           (via adjacent swaps) into one contiguous block (or effectively into one spot)
           is well-known to be the minimum over all consecutive blocks of k ones in the
           sorted positions of those 1s.  In many "minimum swaps to group k ones"
           problems, picking the median of those k positions achieves that minimum.
           
           Concretely, if p[] is the sorted list of indices where nums[] == 1
           (length M), then to pick out a consecutive subwindow p[i.. i+(k-1)] of length k,
           the cost of bringing those k ones to (the median of those positions) is
             sum_{j=i..i+k-1} |p[j] - p[ i + (k//2) ]| 
           and taking the minimum over i gives the "no-creation" cost for choosing exactly k from
           the existing 1s.

        2) If we allow creation of new 1s (up to maxChanges), each newly created 1 costs:
              1 move to create it at some zero position j != aliceIndex,
              + (some number of adjacent swaps to move it to the pickup index).
           However, if we think in simplest terms, placing a new 1 close to the final "center"
           generally costs 1 (creation) + distance swaps.  In the best case (placing it just
           one step away), that costs 2 moves per newly created 1.

           A crucial simplification/standard result is:
           "To pick exactly b from the existing array-positions (by adjacency swaps) plus
            (k - b) created ones, the total cost is:
               costNoCreation(b) + 2 * (k - b),
            as long as (k - b) <= maxChanges.

           Why "+ 2*(k-b)"?  Because creating each new 1 typically costs 1 move, plus at least
           1 swap to pick it up (in the best/adjacent scenario), so 2 moves per created 1.

        3) Therefore, the final answer is found by:
             1) Precompute costNoCreation(b) for b = 0..M  (0 means picking none of the original 1s).
             2) Then for each b in [max(0, k-maxChanges) .. min(k, M)], the candidate cost is
                  costNoCreation(b) + 2*(k - b).
             3) Also, if k <= maxChanges, we can create all k (b=0) => cost = 2*k.
             4) Take the minimum of these.

        Computing costNoCreation(b):
        ----------------------------------------------------------------------------
        - Let p[] be the sorted positions of 1s, length M.
        - costNoCreation(b) = the minimum over all i of
            sum_{j=i..i+b-1} |p[j] - p[i + b//2]|   (the median is p[i + b//2])
          Because to move those b ones together (in a "bubble-sort" sense with 1-0 swaps),
          the cost matches the sum of distances to that median position.
        
        - Implementation detail: we can do this in O(M) time for a fixed b by a known prefix-sum
          trick for absolute deviations from the median:
              let mid = i + (b//2)
              pivot = p[mid]
              leftPart = pivot*(mid-i) - (prefixSumOfP[mid] - prefixSumOfP[i])
              rightPart= (prefixSumOfP[i+b] - prefixSumOfP[mid+1]) - pivot*((i+b-1) - mid)
              cost = leftPart + rightPart
          Then we take the min cost over i in [0..M-b].  That yields costNoCreation(b).

        - We then repeat for b = 1..M.  costNoCreation(0)=0 by definition (picking none of the
          existing 1s => no swap cost).

        Complexity considerations:
        ----------------------------------------------------------------------------
        - Let M = number of 1s in nums.
        - In the worst case M can be up to n (if all entries are 1).
        - For each b, computing costNoCreation(b) takes O(M).  Doing this for up to M values of b
          is O(M^2) in the worst case.  That can be up to 10^10 if M=10^5, which is quite large in
          Python.  In practice, careful (and possibly low-level / C++-style) optimization might be
          required for the largest inputs.  However, the method is correct.

        We'll implement the direct method here for clarity.  If the input sizes were truly large
        and tightly time-limited, one would need further optimizations (or a lower-level language).

        Steps to implement:
        1. Gather positions p[] of the ones in sorted order.
        2. Build prefix sums P[] of p.
        3. Define a function cost_for_subwindow(i, b) -> cost.
        4. For b=0..M, compute costNoCreation(b).
        5. Among b in [max(0, k-maxChanges).. min(k, M)], the candidate cost = costNoCreation(b) + 2*(k-b).
        6. If k <= maxChanges, also consider the cost of creating all k, i.e. 2*k.
        7. Take the minimum of these candidates.
        8. Return it.

        This matches the examples given in the problem statement.
        """

        import sys
        input_data = sys.stdin.read().strip().split()
        # The driver (if used) might feed input differently; here we assume the framework
        # will call minimumMoves(...) directly.  So this parsing is just a template if needed.
        # We'll just implement the core logic assuming nums, k, maxChanges are provided.
        
        p = []
        for i, val in enumerate(nums):
            if val == 1:
                p.append(i)
        M = len(p)
        
        # Quick edge case: if k == 0 (not in problem constraints, though) => cost = 0
        # Quick edge case: if sum(nums) == 0 => then M=0 => we can only create.
        #   The problem states sum(nums) + maxChanges >= k, so creation is possible if M=0.
        if M == 0:
            # If M=0, costNoCreation(b>0) is impossible. We can only do b=0 + creation if k<=maxChanges.
            # cost = 2*k if k <= maxChanges
            # The statement says: maxChanges + sum(nums) >= k => so if sum(nums)=0 => maxChanges>=k
            # So answer = 2*k
            return 2*k

        # Prefix sums of p
        # P[i] = sum of p[0] + p[1] + ... + p[i-1]
        P = [0]*(M+1)
        for i in range(M):
            P[i+1] = P[i] + p[i]

        # costNoCreation[b] = minimal cost to pick any consecutive subwindow of length b from p
        # b=0 => cost=0 by definition (picking none).
        INF = 10**18
        costNoCreation = [INF]*(M+1)
        costNoCreation[0] = 0

        # If we pick subwindow length == 1, cost is always 0 (one element to itself => distance 0).
        # so costNoCreation[1] = 0 (if M >= 1).
        if M >= 1:
            costNoCreation[1] = 0

        # Compute costNoCreation(b) for b=2..M
        # cost of subwindow p[i..i+b-1] with median at p[mid], mid = i + b//2
        # cost = leftPart + rightPart
        # leftPart = pivot*(mid-i) - (P[mid] - P[i])
        # rightPart = (P[i+b] - P[mid+1]) - pivot*((i+b-1) - mid)
        for b in range(2, M+1):
            min_cost_b = INF
            for i in range(M - b + 1):
                mid = i + (b // 2)
                pivot = p[mid]
                leftPart = pivot*(mid - i) - (P[mid] - P[i])
                rightPart = (P[i+b] - P[mid+1]) - pivot*((i+b-1) - mid)
                cost_sub = leftPart + rightPart
                if cost_sub < min_cost_b:
                    min_cost_b = cost_sub
            costNoCreation[b] = min_cost_b

        # Now try combinations: pick b from the existing 1s, create (k-b) new 1s (each costs 2 moves),
        # but only if (k-b) <= maxChanges and b <= M, b >= 0, etc.
        # Also consider the possibility of creating all k (if k <= maxChanges).
        ans = INF

        # Range of b: we must pick at least (k - maxChanges) from existing if that is >= 0,
        # and at most min(k, M).
        start_b = max(0, k - maxChanges)
        end_b   = min(k, M)   # b can't exceed M or k
        if start_b > end_b:
            # means we cannot pick enough from array to reduce creation to within maxChanges
            # but by constraints sum(nums)+maxChanges >= k, it should still be feasible, e.g. b=M, plus creation
            # but if start_b> end_b and M < k, that implies M < k-maxChanges, contradiction to the problem's precondition.
            # We'll just skip if that mathematically happens; or handle next "if k <= maxChanges" case.
            pass
        else:
            for b in range(start_b, end_b + 1):
                # cost picking b from array = costNoCreation[b]
                # plus 2*(k-b) for creation, as long as k-b <= maxChanges
                c = k - b
                if c <= maxChanges:
                    curr = costNoCreation[b] + 2*c
                    if curr < ans:
                        ans = curr

        # Possibly create all k if k <= maxChanges:
        if k <= maxChanges:
            ans = min(ans, 2*k)

        return ans