class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        """
        We need to find the earliest second s (1 <= s <= m) such that there exists a valid
        schedule of operations in seconds 1..s to:

          1) Decrement some nums[i]'s enough times so that nums[i] == 0,
          2) Mark each index i exactly once at some time t with changeIndices[t] == i
             and nums[i] == 0 at that time,
          3) Use at most one operation (decrement or mark or do-nothing) per second,

        such that by time s, all indices i in 1..n are marked.

        Key observations / approach:
          - An index i can only be marked at seconds t where changeIndices[t] == i.
            Let T[i] = sorted list of all t with changeIndices[t] == i.
            If T[i] is empty for some i, it is impossible to mark that i -> return -1 immediately.
          - We need sum(nums[i]) total decrements, plus n marking operations (one per index).
            Each operation occupies a separate second; hence we need at least sum(nums) + n seconds
            with no idle time. If m < sum(nums) + n, it's impossible -> return -1.
          - Even if m >= sum(nums) + n, we must ensure that for each i we can pick a distinct time
            t_i in T[i] to do the marking, with t_i >= 1..s, and that by (t_i - 1) we have performed
            enough total decrements on i to make nums[i] == 0.
          - A classical scheduling check is:
               - We try a candidate s in [sum(nums)+n .. m].
               - We see if we can "assign" each index i a marking time t_i in T[i] ≤ s and also
                 fit all required nums[i] decrements (in total across indices) into the times
                 before t_i.
               - Because we only have one operation per time, by the time we mark i at t_i,
                 we must have used at least nums[i] operations to decrement i, plus one operation
                 for each previously marked index. In a feasible schedule, if you sum up
                 nums[i] + 1 across indices in an appropriate order, each chosen marking time
                 t_i must be at least the partial sum of all operations used so far plus 1 for i.

            Checking feasibility for a given s:
              1) Sort indices by descending nums[i] (largest "work" first).
              2) Keep a running "partialSum" = total operations used so far (decrements + marks).
              3) For each index i:
                 - We need partialSum + nums[i] + 1 <= s to even fit i's work by time s.
                 - Among T[i] ∩ [1..s], we must pick a time t >= partialSum + nums[i] + 1
                   (since we need at least nums[i] decrements + 1 mark before or at second t).
                 - We pick the smallest such t in T[i] that is unused. If none exists, fail.
                 - Mark t as used, update partialSum += nums[i] + 1.
              4) If we succeed for all i, we have a valid schedule by time s.

            Then we pick the earliest s by either a simple linear search from sum(nums)+n..m
            or a binary search in that range.  Here we do a binary search for efficiency.

        Complexity considerations:
          - n, m <= 2000.  sum(nums) can be large (up to n*10^9), but we only need it
            to compute lower bound = sum(nums) + n for s.
          - We do a binary search on s in [sum(nums)+n..m], up to ~2000 steps in the worst case,
            which is about log(2000) ~ 11 steps if we narrow by binary search.  
          - For each feasibility check, we do up to n picks, each requiring a search
            in T[i].  T[i] can be up to length m.  We can do a binary search for a suitable
            time plus a short linear scan to find an unused slot.  That is O(n * (log m + possible scan)).
          - This will pass within reasonable time if implemented carefully in Python.

        Let's implement.
        """
        import bisect

        n = len(nums)
        m = len(changeIndices)

        # Build the list of times T[i] where index i can be marked (1-based to 1-based).
        # i in [1..n], but Python is 0-based; let's store T[i-1] for i in [1..n].
        T = [[] for _ in range(n)]
        for t, idx in enumerate(changeIndices, start=1):
            T[idx - 1].append(t)  # reduce idx by 1 for 0-based

        # If any T[i] is empty, we can never mark i -> return -1
        for i in range(n):
            if len(T[i]) == 0:
                return -1

        # sum of all nums[i]
        total_decrements = sum(nums)
        # The absolute minimum time needed is total_decrements + n (no idle time).
        if total_decrements + n > m:
            return -1  # not enough total seconds even if we do nothing else.

        # Sort each T[i] so we can binary-search in it
        for i in range(n):
            T[i].sort()

        # Function to check feasibility if we only have s seconds (1..s).
        # We'll do a "largest nums[i] first" approach (descending order).
        # We'll try to pick the earliest marking time t ∈ T[i], t <= s, t >= needed,
        # that hasn't been used yet. If we can do so for all i, it's feasible.
        def can_do_by(s: int) -> bool:
            used = [False] * (s + 1)  # track used marking times from 1..s
            # Sort indices by descending nums[i]
            idxs = sorted(range(n), key=lambda x: nums[x], reverse=True)

            partial_sum = 0
            for i in idxs:
                needed = partial_sum + nums[i] + 1  # we must place i's mark at or after this time
                if needed > s:
                    return False  # not enough total time
                # We want to find the smallest t in T[i] with t >= needed and t <= s, and not used
                arr = T[i]  # sorted list of possible times
                pos = bisect.bisect_left(arr, needed)
                # Now linearly scan forward while t is in range and is used
                while pos < len(arr) and arr[pos] <= s:
                    if not used[arr[pos]]:
                        # we can use this time
                        used[arr[pos]] = True
                        break
                    pos += 1
                else:
                    # no suitable t found
                    return False
                # Update partial_sum: we used nums[i] decrements + 1 mark
                partial_sum += nums[i] + 1

            return True

        # We'll do a binary search on s in [total_decrements + n .. m]
        left = total_decrements + n
        right = m
        answer = -1

        while left <= right:
            mid = (left + right) // 2
            if can_do_by(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1

        return answer