class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        
        for i in range(n):
            min_val = nums[i]
            max_val = nums[i]
            
            for j in range(i, n):
                min_val = min(min_val, nums[j])
                max_val = max(max_val, nums[j])
                
                if max_val - min_val <= 2:
                    count += 1
                else:
                    break
        
        return count