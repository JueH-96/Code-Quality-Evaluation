class Solution:
    def minDifference(self, nums: List[int]) -> int:
        D_fixed = 0
        n = len(nums)
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                diff = abs(nums[i+1] - nums[i])
                if diff > D_fixed:
                    D_fixed = diff
        
        known_indices = [i for i, num in enumerate(nums) if num != -1]
        max_required_D = 0
        
        for i in range(len(known_indices) - 1):
            prev_idx = known_indices[i]
            curr_idx = known_indices[i+1]
            L = nums[prev_idx]
            R = nums[curr_idx]
            m = curr_idx - prev_idx - 1
            if m >= 1:
                distance = abs(R - L)
                if m == 1:
                    required_D = (distance + 1) // 2
                else:
                    required_D = (distance + 2) // 3
                if required_D > max_required_D:
                    max_required_D = required_D
        
        if not known_indices:
            return 0
        else:
            return max(D_fixed, max_required_D)