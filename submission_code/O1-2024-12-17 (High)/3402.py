class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        """
        We want to make all elements in nums equal using two possible increment operations:
          1) Increase one element by 1 at a cost of cost1.
          2) Increase two distinct elements by 1 at a cost of cost2.

        We seek the minimum total cost to achieve equality, modulo 1e9+7.

        Key observations and approach:

        1) Let n = len(nums), S = sum(nums), minVal = min(nums), maxVal = max(nums).
           We must choose a final value T >= maxVal (since we cannot decrease any number).

        2) Define K(T) = sum( (T - nums[i]) ) = n*T - S  (total increments needed)
           and let Dmax(T) = T - minVal (the largest single-element gap).
           If cost2 >= 2*cost1, it is never cheaper to use the "two-increments" operation;
           in that case, the best choice is T = maxVal, and cost = cost1 * sum(maxVal - nums[i]).

        3) If cost2 < 2*cost1, we can benefit from pairing increments.  
           In an ideal scenario, the maximum number of 2-increment operations (pairs) is
             p(T) = min( floor(K(T)/2), K(T) - Dmax(T) ).
           Explanation: 
             - floor(K/2) is the absolute maximum pairs if you could pair every increment.  
             - K - Dmax(T) is how many increments can be paired before the element needing
               the largest gap runs out of partners.
           Then leftover = K(T) - 2*p(T).  
           The total cost = p(T)*cost2 + leftover*cost1.

        4) Critically, T can sometimes be larger than maxVal to allow more pairing, which may
           reduce cost if cost2 is sufficiently cheap.  However, pushing T too far eventually
           becomes counter-productive because K(T) grows linearly with T.

           Empirically (and by examples), one can show that the optimal T will lie in
           the integer interval [maxVal, maxVal + (maxVal - minVal)].
           Beyond T = maxVal + (maxVal - minVal), costs begin to grow again.

        5) Therefore:
           - If all elements are already equal (maxVal == minVal), cost = 0.
           - If cost2 >= 2*cost1, use T = maxVal with single-increment operations only.
           - Otherwise, iterate T in [maxVal .. maxVal + (maxVal - minVal)] to find the
             minimal cost via the pairing formula.  This range is at most about 10^6 in length
             (since nums[i] <= 10^6), which is feasible in Python if done carefully.

        6) Return the minimal cost found, modulo 1e9+7.
        """

        mod = 10**9 + 7
        n = len(nums)
        if n <= 1:
            # If there's only one element, it's already "equal"; cost = 0
            return 0

        min_val = min(nums)
        max_val = max(nums)
        if min_val == max_val:
            # Already all equal
            return 0

        # Sum of all elements
        S = sum(nums)

        # If it's never cheaper to do the two-element increment,
        # just raise everyone to max_val individually:
        if cost2 >= 2*cost1:
            single_increments = 0
            for x in nums:
                single_increments += (max_val - x)
            # Cost:
            return (single_increments * cost1) % mod

        # Otherwise, we try all T in [max_val .. max_val + (max_val - min_val)]
        # and pick the minimal cost among them.
        diff = max_val - min_val
        best_cost = None

        # Precompute this once outside the loop:
        base = n * max_val - S  # This is K(max_val).

        # We'll iterate from T = max_val up to T = max_val + diff.
        # Each time T increases by 1, K(T) = n*(T) - S = K(T-1) + n.
        # Also, (T - min_val) increases by 1 each time.
        #
        # So we can update these values incrementally instead of recomputing from scratch.
        K = base           # This is n*max_val - S when T = max_val
        largest_gap = max_val - min_val  # T-min_val when T = max_val

        for add in range(diff + 1):
            # Now T = max_val + add
            # K = n*T - S
            # largest_gap = (max_val + add) - min_val

            # Number of pair-increments we can do:
            # p = min( K//2, K - largest_gap ) but not below 0
            p_candidate1 = K // 2
            p_candidate2 = K - largest_gap
            if p_candidate2 < 0:
                p_candidate2 = 0
            p = min(p_candidate1, p_candidate2)

            leftover = K - 2*p
            cost = p*cost2 + leftover*cost1

            if best_cost is None or cost < best_cost:
                best_cost = cost

            # Increment K and largest_gap for the next T
            K += n
            largest_gap += 1

        return best_cost % mod