from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # prefix_max[i] will store max(nums[0..i])
        prefix_max = [0] * n
        prefix_max[0] = nums[0]
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i - 1], nums[i])

        # suffix_max[i] will store max(nums[i+1 .. end])
        suffix_max = [float('-inf')] * n
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(suffix_max[i + 1], nums[i + 1])

        best = 0
        # j must be between 1 and n-2
        for j in range(1, n - 1):
            diff = prefix_max[j - 1] - nums[j]      # best (nums[i] - nums[j]) with i < j
            if diff <= 0:                            # product will be non-positive
                continue
            candidate = diff * suffix_max[j]         # multiply by best nums[k] with k > j
            best = max(best, candidate)

        return best