from typing import List
from collections import deque

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Queue to store the number of operations started at past indices j
        # whose effect is still active at the current index i.
        # Specifically, q stores ops_at[j] for j such that max(0, i-k+1) <= j < i
        # (conceptually, these are the operations that started at index j
        # and affect the current index i).
        # The size of the queue will be at most k-1.
        q = deque()
        
        # window_sum tracks the total decrease applied to the current index `i`
        # by operations that started at indices j such that max(0, i-k+1) <= j < i.
        window_sum = 0

        for i in range(n):
            # Remove the effect of operations that started k steps ago.
            # These operations no longer affect the current index i.
            # The operation starting at index i-k (if i-k >= 0) affects indices
            # from i-k to i-k+k-1 = i-1. So, at index i, its effect ends.
            # The number of operations started at i-k was added to the queue
            # when processing index i-k. Now, k steps later, it's at the front.
            if i >= k:
                # q.popleft() gives the number of operations started at index i-k.
                window_sum -= q.popleft()

            # Calculate the remaining value at nums[i] after applying previous operations.
            # This remaining value must be covered by operations starting at index i.
            # 'needed' represents the required number of operations that must start at index i.
            needed = nums[i] - window_sum

            # If needed < 0, it means previous operations over-reduced nums[i]. Impossible.
            if needed < 0:
                return False

            # If needed > 0, we must apply 'needed' operations starting at index i.
            # These operations are only valid if the subarray [i, i+k-1] is within bounds.
            # If needed > 0 and i + k > n, we cannot start a full subarray operation of size k at index i.
            # This means nums[i] is still positive and cannot be reduced further by valid ops. Impossible.
            if needed > 0 and i + k > n:
                return False
            
            # If needed is non-negative and we can start operations at index i (i.e., i + k <= n):
            # The number of ops started at i is 'needed'. This amount contributes to the window_sum
            # for future indices from i+1 up to i+k-1. We add 'needed' to the queue
            # (to be removed after k steps) and update window_sum.
            # This handles both needed > 0 and needed == 0 cases when i + k <= n.
            if i + k <= n:
                q.append(needed)
                window_sum += needed

            # If needed == 0 and i + k > n, it's fine. nums[i] is already zeroed by previous ops.
            # Index i is not a valid starting position for a size k op, so we don't append to queue.
            # This case is implicitly handled by the structure:
            # if needed == 0, the first check `if needed > 0 and i + k > n` passes.
            # if i + k > n, the `if i + k <= n` block is skipped.
            # So, when needed == 0 and i + k > n, nothing happens, which is correct.

        # If the loop finishes without returning False, it means all elements could be reduced to 0.
        return True