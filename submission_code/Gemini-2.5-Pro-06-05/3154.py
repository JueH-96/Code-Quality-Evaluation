from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        """
        Calculates the maximum value of (nums[i] - nums[j]) * nums[k] for i < j < k.

        This method uses a single-pass O(n) approach to find the maximum triplet value.
        It iterates through the array, maintaining the maximum element seen so far (`max_prefix`)
        and the maximum difference `(nums[i] - nums[j])` found so far (`max_diff`).

        For each element `num` in the array (considered as `nums[k]`):
        1. A potential maximum value is calculated as `max_diff * num`. The overall answer `ans` is updated.
        2. The state is updated for future iterations:
           - `max_diff` is updated by considering the current `num` as a potential `nums[j]`.
           - `max_prefix` is updated by considering the current `num` as a potential `nums[i]`.
        """
        ans = 0
        
        # max_prefix will store the maximum value of nums[i] seen so far.
        max_prefix = 0
        
        # max_diff will store the maximum difference (nums[i] - nums[j]) seen so far.
        max_diff = 0
        
        for num in nums:
            # At this point, the current `num` acts as nums[k].
            # We calculate a potential answer using the max_diff found from the prefix
            # of the array before the current element.
            ans = max(ans, max_diff * num)
            
            # Now, we update the state for future iterations.
            
            # The current `num` can act as a future `nums[j]`. The best `nums[i]`
            # to pair it with is `max_prefix` (the max element seen before `num`).
            # We update `max_diff` with this new possibility.
            max_diff = max(max_diff, max_prefix - num)
            
            # The current `num` also updates the `max_prefix` for future `nums[i]` candidates.
            max_prefix = max(max_prefix, num)
            
        return ans