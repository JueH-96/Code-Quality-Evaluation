from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        """
        We have n indices (1..n) and an array nums of length n (1-indexed in the problem).
        We also have m seconds (1..m) and an array changeIndices of length m giving which
        index could be marked at each second (1-indexed in the problem).

        In each of the m seconds (in order), we are allowed exactly one of:
          (a) decrement nums[i] by 1 for some i, or
          (b) if nums[changeIndices[s]] == 0, mark that index, or
          (c) do nothing.

        The goal is to find the earliest second s in [1..m] such that there is a way
        (an optimal schedule of operations) to get all indices marked by or at second s.
        Return -1 if it's impossible.

        ---------------------------------------------------------
        IDEA AND APPROACH:

        1) Key Insights:
           - Each index i can only be marked at seconds s where changeIndices[s] == i.
           - Marking i requires that nums[i] has been decremented down to 0 strictly *before* we use
             second s to mark (since we can only do one operation at second s: the mark itself).
           - Altogether, for each i, we need nums[i] "decrement" operations plus 1 "mark" operation.
             So index i effectively needs (nums[i] + 1) operations in total.
           - We have at most m seconds, each second can perform exactly one operation, thus a quick
             necessary condition is sum(nums) + n ≤ m.

        2) Checking feasibility by a given time T:
           We ask: "Can we complete all required operations (sum(nums[i]) + n of them) by second T
           and also ensure that each index i is marked at some valid second s ∈ S_i (where S_i are
           the seconds s for which changeIndices[s] = i) with s ≤ T?"

           Conditions:
             - We cannot mark i at time s unless nums[i] is already 0 by the end of second (s-1).
             - We also cannot use the same second s for two different operations (no overlap).
             - Each index i must pick exactly one mark-time s_i in S_i (with s_i ≤ T).
               Then we must place nums[i] decrements strictly before s_i in distinct seconds.

           A Greedy Scheduling Method (checking feasibility):
             - First, if sum(nums) + n > T, not feasible (not enough time-slots).
             - For each i, let S'_i = { s ∈ S_i : s ≤ T and s-1 ≥ nums[i] }, because
               we can only mark i at time s if we have at least nums[i] slots before s
               to decrement i. If S'_i is empty for any i, not feasible.
             - Then we try to assign slots:
               Sort indices i in descending order of nums[i] (largest decrement-need first).
               Keep a list (or set) of free slots = {1, 2, ..., T}.
               For each i in that sorted order:
                 - Examine S'_i in descending order (largest possible mark-time first).
                 - If that candidate mark-time s is free, check if there are ≥ nums[i] free slots
                   among {1, 2, ..., s-1}. If yes, assign exactly nums[i] of those free slots to
                   i's decrements (pick any of them, typically the largest ones < s). Then mark
                   slot s as used for "marking i". Break out and proceed to the next i.
                 - If no valid s in S'_i works, this T is not feasible.
               If we succeed for all indices, T is feasible.

           We wrap this in a (binary) search over T from (sum(nums)+n) up to m to find the earliest T.

        3) Complexity:
           - n, m ≤ 2000, sum(nums) ≤ 2000 (otherwise it's outright impossible).
           - We do a binary search over T in [sum(nums)+n .. m], which is at most ~log(m) checks.
           - Each feasibility check is O(n*T + sum(nums)) in a careful implementation.
           - This is acceptable for n, T up to 2000 in a well-optimized Python solution.

        Let's implement it.
        """

        n = len(nums)
        m = len(changeIndices)

        # Precompute all positions for each index i (0-based).  positions[i] will hold
        # all times s (1-based) such that changeIndices[s-1] == i+1 in ascending order.
        positions = [[] for _ in range(n)]
        for s, idx in enumerate(changeIndices):
            positions[idx - 1].append(s + 1)

        # Quick check: if there's an i with no occurrence in changeIndices but nums[i] >= 0,
        # we cannot ever mark it => immediately -1.
        # (Even if nums[i] = 0, we still need a time to mark it, which doesn't exist.)
        for i in range(n):
            if not positions[i]:  # empty
                return -1

        sumA = sum(nums)
        # If sumA + n > m, impossible from the start.
        if sumA + n > m:
            return -1

        # Define a function to check feasibility for a given T.
        def can_finish_by(T: int) -> bool:
            # 1) Quick check of total operations
            if sumA + n > T:
                return False

            # 2) Build the feasible mark-times S'_i in descending order for each i
            feasible_times = []
            for i in range(n):
                # Filter positions[i] to those <= T and also s-1 >= nums[i]
                needed = nums[i]
                cand = []
                for s in positions[i]:
                    if s <= T and (s - 1) >= needed:
                        cand.append(s)
                # If no valid s, fail
                if not cand:
                    return False
                # Sort descending
                cand.sort(reverse=True)
                feasible_times.append(cand)

            # 3) Sort indices by descending nums[i] so we place the "largest demand" first
            idx_order = sorted(range(n), key=lambda x: nums[x], reverse=True)

            # 4) We'll keep track of which seconds are free for operations
            #    True = free, False = used
            #    We'll need a prefix sum approach to find how many free are < s quickly,
            #    plus to actually pick them from right to left if enough are available.
            free_slot = [True] * (T + 1)  # 1..T used, index 0 unused

            # We'll prepare a small helper for counting free slots up to x:
            # We'll just re-build a prefix each time we try to place one index's tasks
            # because T <= 2000 is not too large.

            for i in idx_order:
                need = nums[i]
                found_spot = False
                for s in feasible_times[i]:
                    if not free_slot[s]:
                        continue  # marking time s is already used

                    # Count how many free slots are in [1..s-1]
                    # and if it's >= need, try to assign
                    count_free = 0
                    for x in range(1, s):
                        if free_slot[x]:
                            count_free += 1
                    if count_free < need:
                        continue

                    # We can place i's decrements in the largest 'need' free slots below s
                    to_use = []
                    c = need
                    # pick from s-1 down to 1
                    x = s - 1
                    while c > 0 and x >= 1:
                        if free_slot[x]:
                            to_use.append(x)
                            c -= 1
                        x -= 1

                    # Mark them used
                    for slot in to_use:
                        free_slot[slot] = False
                    # Mark s itself for the "mark" operation
                    free_slot[s] = False
                    found_spot = True
                    break

                if not found_spot:
                    return False

            return True

        # We'll do a binary search over T in [sumA+n .. m] to find earliest feasible T
        lo = sumA + n
        hi = m
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_finish_by(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return ans