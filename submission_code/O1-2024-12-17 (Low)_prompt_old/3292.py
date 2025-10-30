class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        """
        We want the earliest second s in [1..m] for which we can schedule:
          • sum(nums) total decrements (one-at-a-time)
          • n markings (one for each index)
        subject to:
          • We can perform exactly one operation per second (either a decrement, or marking if the
            referenced index in changeIndices[s] is already 0, or do nothing).
          • We can only mark index i at second t if changeIndices[t] == i and nums[i] == 0 by that time.

        A necessary condition is that sum(nums)+n <= m (we need at least enough seconds to do all required
        operations if we ignore the constraint about which index can be marked in each second). However, the
        order also matters: each index i must appear in changeIndices in some second we haven't used for a
        different mark, after enough decrements have been spent to reduce nums[i] to zero.

        We use a binary search for the earliest second s in [1..m], then check feasibility as follows:

          check(s):
            1) If sum(nums) + n > s, it's impossible (not enough total operations).
            2) For each i, let T_i = sorted list of all seconds t where changeIndices[t] == i.
               If T_i has no t <= s, we cannot mark i within the first s seconds → fail.
            3) We must pick exactly one time x_i ∈ T_i (and x_i ≤ s) to mark i. One operation per second,
               so if we have already chosen j-1 marks strictly before x_i, that means there are (x_i - 1) - (j-1)
               = x_i - j seconds left for decrements among the first x_i - 1 slots. By the time we do the j-th mark,
               the total number of decrements used for all j indices so far must not exceed (x_i - j).
               In particular, if “sumDecrSoFar” is the total of nums for indices we have assigned times so far, 
               and we want to assign index i with nums[i], we need:
                     x_i - j >= sumDecrSoFar + nums[i].
            4) We do the assignment in descending order of nums[i]. For index i, we try to pick the smallest
               time t in T_i (t ≤ s) that satisfies t - j >= sumDecrSoFar + nums[i]. If no such t exists, fail.
               Otherwise, fix that t as the marking time for i, and sumDecrSoFar += nums[i], j += 1.
            5) If we succeed in assigning marking times for all i, then it is feasible to finish by second s.

        We binary search over s = 1..m to find the smallest s that passes check(s). If none pass, return -1.
        """

        import bisect

        n = len(nums)
        m = len(changeIndices)
        # Precompute the list of times T_i where changeIndices[time-1] = i (1-based => time-1 for array)
        # T_i will be sorted ascending.
        positions = [[] for _ in range(n)]
        for t, i in enumerate(changeIndices, start=1):
            # i is 1-indexed, we convert to 0-based
            positions[i-1].append(t)

        # If any index i never appears in changeIndices, it's impossible to mark it.
        for i in range(n):
            if not positions[i]:
                # If nums[i] > 0 or we still need to mark i even if nums[i] == 0, it's impossible.
                # Actually, even if nums[i] = 0, we still must mark it, so we need i in changeIndices.
                return -1

        # Total operations needed = sum(nums) decrements + n markings
        total_decrements = sum(nums)
        if total_decrements + n > m:
            return -1  # Not enough total seconds in any case

        # Binary search over s = [1..m]
        def can_finish_by(s):
            # Quick check if we even have enough time slots in [1..s].
            if total_decrements + n > s:
                return False

            # We'll try to assign a valid marking time t_i <= s for each index i (0-based),
            # in descending order of nums[i]. For i, we need a time t such that
            # t - (already_assigned_marks+1) >= (sum_decr_so_far + nums[i]).
            # Because at second t we do the mark (the (already_assigned_marks+1)-th mark),
            # so the first t-1 seconds can have at most (t-1)-(already_assigned_marks) = t-1 - already_assigned_marks
            # = (t - (already_assigned_marks+1)) decrements in them. We want that to be >= sum_decr_so_far+nums[i].
            #
            # We'll pick the smallest t in positions[i] that is <= s and satisfies the condition above.
            # If none exist, return False.

            # Sort indices by descending nums[i] (largest first).
            idxs = sorted(range(n), key=lambda x: nums[x], reverse=True)

            sum_decr = 0  # sum of decrements for the indices assigned so far
            marks_used = 0
            # For each index i in that order, find a feasible marking time.
            for i in idxs:
                needed_decr = nums[i]
                # We need t - (marks_used+1) >= sum_decr + needed_decr
                # => t >= sum_decr + needed_decr + marks_used + 1
                needed_t = sum_decr + needed_decr + (marks_used + 1)

                # Among positions[i] that are <= s, do a binary search for the smallest t >= needed_t
                # i.e. we want positions[i][k] >= needed_t and positions[i][k] <= s.
                # We'll do a pointer or bisect.
                arr = positions[i]
                # We want to find left = bisect_left(arr, needed_t)
                left = bisect.bisect_left(arr, needed_t)
                # Now we check if arr[left] is within s
                if left == len(arr):
                    # no valid time (all times < needed_t)
                    return False
                t_candidate = arr[left]
                if t_candidate > s:
                    return False

                # If it is valid, we "use" t_candidate for marking index i
                # update sum_decr, marks_used
                sum_decr += needed_decr
                marks_used += 1

            return True

            # If we manage to assign times for all indices, it means we can do it by second s.

        # Binary search
        left, right = 1, m
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if can_finish_by(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans