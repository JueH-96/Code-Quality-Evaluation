class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            max_elem = nums[i]
            for j in range(i, n):
                max_elem = max(max_elem, nums[j])
                if nums[i] == max_elem and nums[j] == max_elem:
                    count += 1
        
        return count