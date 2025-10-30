from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        """
        Finds the earliest second to mark all indices in nums.

        This problem can be solved using binary search on the answer. The core idea is that
        if it's possible to mark all indices in `k` seconds, it's also possible in `k+1`
        seconds. This monotonic property makes binary search applicable on the number of
        seconds, `k`, from 1 to `m`.

        The main challenge is to implement a `check(k)` function that efficiently determines
        if it's possible to mark all indices within `k` seconds. A greedy strategy works
        best here: for each index `i`, we should plan to mark it at its last possible
        opportunity within the first `k` seconds. This maximizes the number of preceding
        "free" seconds available for decrementing `nums[i]` to zero.

        The `check(k)` function proceeds as follows:
        1. Find the last second (from 1 to k) where each index `i` can be marked.
        2. If any index cannot be marked within `k` seconds, it's impossible.
        3. Simulate the process from second 1 to `k`. We maintain a count of "free slots"
           (seconds not used for marking). On a designated marking day for an index `i`,
           we check if we have accumulated enough free slots to cover the cost of
           decrementing `nums[i]`. If not, the check fails. Otherwise, we use up the
           required slots. On any other day, we gain a free slot.
        4. If the simulation completes, the check passes.

        The main function binary searches for the smallest `k` for which `check(k)` is true.
        """
        n = len(nums)
        m = len(changeIndices)

        # The problem uses 1-based indexing. Convert changeIndices values to 0-based.
        changeIndices_0based = [c - 1 for c in changeIndices]

        def check(k: int) -> bool:
            # k is the 1-based number of seconds, so we consider seconds 0 to k-1.
            
            # Step 1: Find the last second each index can be marked.
            last_pos = {}
            for i in range(k):
                last_pos[changeIndices_0based[i]] = i
            
            # If not all n indices appear in the first k seconds, it's impossible.
            if len(last_pos) != n:
                return False

            # Step 2 & 3: Simulate to check if decrement costs can be met.
            free_slots = 0
            for i in range(k):
                idx_at_this_second = changeIndices_0based[i]
                
                # Check if this second is the designated marking day for the index.
                if i == last_pos.get(idx_at_this_second):
                    cost = nums[idx_at_this_second]
                    # Do we have enough free slots to decrement this number to 0?
                    if free_slots < cost:
                        return False
                    free_slots -= cost
                else:
                    # This second is not a designated marking day, so it's a free slot
                    # that can be used for a decrement operation.
                    free_slots += 1
            
            return True

        # Binary search for the earliest second k.
        low = 1
        high = m
        ans = -1

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans