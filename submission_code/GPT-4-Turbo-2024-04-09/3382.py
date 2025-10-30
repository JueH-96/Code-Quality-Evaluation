class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        
        for i in range(n):
            max_val = nums[i]
            # Explore all subarrays starting from index i
            for j in range(i, n):
                max_val = max(max_val, nums[j])
                # Check if the current subarray [i:j+1] satisfies the condition
                if nums[i] == nums[j] == max_val:
                    count += 1
        
        return count