from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        # Convert changeIndices to 0-indexed
        ci = [x - 1 for x in changeIndices]

        def can(T: int) -> bool:
            # T is the number of seconds (1-indexed). We use seconds 0 to T-1.
            if T < n:
                return False

            # Find the last occurrence of each index in changeIndices[0...T-1]
            last_s = [-1] * n
            for s in range(T):
                idx = ci[s]
                last_s[idx] = s

            # If any index does not appear in the first T seconds, it's impossible to mark it.
            if -1 in last_s:
                return False

            # Greedily determine which seconds MUST be used for marking.
            # Process seconds from T-1 down to 0. If a second 's' is the latest opportunity
            # to mark index ci[s] among the indices that still need a designated mark time,
            # then second 's' MUST be used for marking index ci[s]. These seconds cannot
            # be used for decrementing. All other seconds can be used for decrementing.

            indices_assigned_mark_time = set()
            decrement_ops_available = 0

            # Iterate through seconds from T-1 down to 0
            for s in range(T - 1, -1, -1):
                idx = ci[s]

                if idx not in indices_assigned_mark_time:
                    # This second s is the latest opportunity to mark index idx within 0...T-1
                    # that we haven't processed yet from the end. This effectively means s == last_s[idx].
                    # We MUST use this second s for marking index idx by time T.
                    indices_assigned_mark_time.add(idx)
                    # This second s is used for marking. It cannot be used for decrementing.
                else:
                    # This second s is not the latest opportunity for index idx (its last_s is > s).
                    # This second can be used for decrementing *any* index.
                    decrement_ops_available += 1

            # If we couldn't find a latest opportunity for all n indices within [0, T-1], impossible.
            # This check is implicitly covered by the `if -1 in last_s` check, because if
            # `last_s[i] == -1` for some i, then i will never be in `indices_assigned_mark_time`
            # after the loop finishes, and `len(indices_assigned_mark_time)` would be less than `n`.
            # if len(indices_assigned_mark_time) < n:
            #    return False

            # Calculate total required decrement operations
            total_needed_decrements = sum(nums)

            # Check if we have enough available seconds (those not used for mandatory last-chance marking)
            # to perform all necessary decrement operations.
            return decrement_ops_available >= total_needed_decrements

        # Binary search for the earliest time T
        # The minimum possible time is n (need n mark operations, 1 sec each).
        # The maximum possible time is m.
        low = 1
        high = m
        ans = -1

        while low <= high:
            mid = (low + high) // 2
            if can(mid):
                ans = mid
                high = mid - 1 # Try earlier time
            else:
                low = mid + 1 # Need more time

        return ans