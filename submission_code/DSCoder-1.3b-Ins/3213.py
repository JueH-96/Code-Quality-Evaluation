class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left, right = -1, -1
        k_index = -1
        for i in range(n):
            if nums[i] == k:
                k_index = i
                break
        
        if k_index == -1:
            return 0
        
        count = 0
        for i in range(k_index, -1, -1):
            if nums[i] >= k:
                right = i
                break
        
        for i in range(k_index, n):
            if nums[i] >= k:
                left = i
                break
        
        if right == -1 or left == -1:
            return 0
        
        count += (right - k_index + 1) * (k_index - left)
        count += (k_index - left) * (right - k_index + 1)
        count += (k_index - left) * (right - k_index)
        
        return count