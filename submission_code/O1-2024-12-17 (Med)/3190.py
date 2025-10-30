class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # For convenience, let:
        #   a = nums1[n-1]  (the "would-be" maximum of final nums1 if we do NOT swap at index n-1)
        #   b = nums2[n-1]  (the "would-be" maximum of final nums2 if we do NOT swap at index n-1)
        #
        # We will consider two scenarios:
        #
        # Scenario A (no swap at n-1):
        #   Final nums1[n-1] = a, Final nums2[n-1] = b
        #   So we want "a" to be the maximum of final nums1, and "b" to be the maximum of final nums2.
        #   This requires that for each i in [0..n-2], the final nums1[i] <= a, final nums2[i] <= b.
        #   We can achieve final nums1[i] and nums2[i] by either swapping or not swapping at index i.
        #
        # Scenario B (swap at n-1):
        #   Final nums1[n-1] = b, Final nums2[n-1] = a
        #   So we want "b" to be the maximum of final nums1, and "a" to be the maximum of final nums2.
        #   This requires that for each i in [0..n-2], the final nums1[i] <= b, final nums2[i] <= a.
        #   Additionally, we pay +1 swap cost for the forced swap at index n-1.
        #
        # In either scenario, we check feasibility index by index for [0..n-2]:
        #   no_swap_ok = (nums1[i] <= M1) and (nums2[i] <= M2)
        #   swap_ok    = (nums2[i] <= M1) and (nums1[i] <= M2)
        #   If neither is true -> infeasible
        #   If both are true -> pick no-swap for zero cost
        #   If exactly one is true -> pick that option (cost 1 if swap_ok is the only choice)
        #
        # We compute the minimal total swaps needed in each scenario and take the minimum.
        # If neither scenario is feasible, return -1.

        a = nums1[n - 1]
        b = nums2[n - 1]

        def feasibility_check(M1, M2):
            """Returns the minimal number of swaps needed (0 or 1 per index) to ensure
               final nums1[i] <= M1 and final nums2[i] <= M2 for i in [0..n-2].
               If impossible, return None."""
            cost = 0
            for i in range(n - 1):
                no_swap_ok = (nums1[i] <= M1 and nums2[i] <= M2)
                swap_ok    = (nums2[i] <= M1 and nums1[i] <= M2)
                if not no_swap_ok and not swap_ok:
                    return None  # infeasible
                if no_swap_ok and not swap_ok:
                    # Must not swap, cost += 0
                    continue
                elif not no_swap_ok and swap_ok:
                    # Must swap, cost += 1
                    cost += 1
                else:
                    # Both ok, pick the cheaper (no swap -> cost 0)
                    continue
            return cost

        # Scenario A: final nums1[n-1] = a, final nums2[n-1] = b
        costA = feasibility_check(a, b)

        # Scenario B: final nums1[n-1] = b, final nums2[n-1] = a, so we add +1 swap at n-1
        costB = feasibility_check(b, a)
        if costB is not None:
            costB += 1

        # Now pick the best feasible scenario
        candidates = []
        if costA is not None:
            candidates.append(costA)
        if costB is not None:
            candidates.append(costB)
        return min(candidates) if candidates else -1