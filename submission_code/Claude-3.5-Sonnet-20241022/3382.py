class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # For each possible subarray starting at i
        for i in range(n):
            curr_max = nums[i]
            # For each possible end point j
            for j in range(i, n):
                # Keep track of max in current window
                curr_max = max(curr_max, nums[j])
                # If first and last elements equal max, increment count
                if nums[i] == curr_max and nums[j] == curr_max:
                    count += 1
                    
        return count