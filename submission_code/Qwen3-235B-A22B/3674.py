from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            current_max = nums[i]
            cost = 0
            for j in range(i, n):
                if j > i:
                    current_max = max(current_max, nums[j])
                # cost += (current_max - nums[j])
                cost += current_max - nums[j]
                if cost <= k:
                    count += 1
                else:
                    # No need to continue this inner loop since further extensions will only increase cost
                    break
        return count