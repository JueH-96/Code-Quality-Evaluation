class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        We want to partition the n indices of nums into some number of groups G, subject to:
         1) Each group is "homogeneous": all indices in a group share the same nums-value.
         2) The difference in group sizes between any two groups is at most 1.

        We seek the smallest such G.

        Observing that:
         - If we fix G, then all group sizes must be either p or p+1, where p = n // G.
           (Because the difference between group sizes is at most 1, so they must come
            from exactly two possible sizes that differ by 1.)
         - If r = n % G, then there will be exactly r groups of size p+1 and (G-r) groups of size p.

        Also, because groups are homogeneous, each value's frequency freq[v] can only go into
        groups of size p or p+1 (i.e. we split freq[v] into some integer number of groups each
        of which is p or p+1 in size). Summing across all values must total exactly G groups
        (r of them are p+1-sized, and G-r of them are p-sized).

        A known necessary-and-sufficient check for feasibility (for a chosen G) is:
         Let p = n // G, r = n % G.

         For each distinct value v with frequency f:
           - it needs at least ceil(f / (p+1)) groups, since each group can hold at most p+1 of that value
             (call that min_groups[v]).
           - it can use at most floor(f / p) groups, since each group must have at least p of that value
             (call that max_groups[v]).
           Define:
              minG_v = the minimum feasible number of groups for freq f, which is ceil(f/(p+1)).
              maxG_v = the maximum feasible number of groups for freq f, which is floor(f/p).

              Then, each chosen number of groups g_v must lie in [minG_v, maxG_v].

              If we let leftover_v = f - g_v * p, that counts how many "extra" items (above p) are needed
              in those g_v groups. Each extra item means one of the g_v groups is of size p+1 instead
              of p. Hence, leftover_v must equal the number of p+1-sized groups among those g_v groups.
              That leftover_v cannot exceed g_v (since each group can only be p+1, so at most 1 extra
              item per group), and leftover_v >= 0.

         Overall feasibility conditions for a chosen G:
           - sum_v minG_v <= G <= sum_v maxG_v      (we can fit a valid number of groups total)
           - Let sum_min_left = Σ( f - maxG_v * p )  (the minimal total leftover if each value uses the maximum number of groups)
             Let sum_max_left = Σ( f - minG_v * p )  (the maximal total leftover if each value uses the minimum number of groups)
             We need sum_min_left <= r <= sum_max_left
             Because r is the total number of size-(p+1) groups we can have, and leftover_v = # of size-(p+1) groups for that value.
           
           If both sets of bounds are satisfied, G is feasible.

        We then do a binary search for G from 1..n to find the minimum feasible G.

        Time complexity is O(m * log n) where m is the number of distinct values (up to n).
        This is acceptable for n <= 10^5.

        Examples:
          nums = [3,2,3,2,3], freq = {3:3, 2:2}, n=5.
          The answer is 2.

          nums = [10,10,10,3,1,1], freq = {10:3,3:1,1:2}, n=6.
          The answer is 4.
        """
        from math import ceil, floor

        # Count frequencies of each distinct value
        from collections import Counter
        freq_counter = Counter(nums)
        freqs = list(freq_counter.values())
        n = len(nums)

        # Helper to check feasibility of a given G
        def can_do(G):
            p = n // G  # base group size
            r = n % G   # how many groups might be of size p+1

            # We want to see if we can distribute the frequencies among exactly G groups,
            # with exactly r groups of size (p+1) and (G-r) groups of size p.
            # We'll compute bounding sums to test feasibility.

            sum_min_groups = 0
            sum_max_groups = 0
            sum_min_left = 0
            sum_max_left = 0

            for f in freqs:
                # If p=0, that would mean G>n, but we will not call can_do with G>n in the search.
                # So we assume p>=1 here.

                # min groups needed (each group can hold up to p+1):
                min_g = (f + (p+1) - 1) // (p+1)  # ceil(f/(p+1))
                # max groups possible (each group holds at least p):
                # if f < p, this might be 0 (not feasible if min_g>0).
                max_g = f // p

                # If min_g > max_g, it's outright impossible
                if min_g > max_g:
                    return False

                sum_min_groups += min_g
                sum_max_groups += max_g

                # minimal leftover occurs if we use as many groups as possible = max_g
                # leftover then is f - max_g * p
                # maximal leftover occurs if we use as few groups as possible = min_g
                # leftover = f - min_g * p
                sum_min_left += f - max_g * p
                sum_max_left += f - min_g * p

            # We need sum_min_groups <= G <= sum_max_groups
            if sum_min_groups > G or G > sum_max_groups:
                return False

            # We need sum_min_left <= r <= sum_max_left
            if sum_min_left > r or r > sum_max_left:
                return False

            return True

        # Binary search for the minimum G in [1..n]
        left, right = 1, n
        ans = n  # worst-case: each index in its own group
        while left <= right:
            mid = (left + right) // 2
            if can_do(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans