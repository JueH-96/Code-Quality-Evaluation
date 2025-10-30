class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Check each possible subarray [i, j]
        for i in range(n):
            for j in range(i, n):
                # Check if removing subarray [i, j] makes array strictly increasing
                is_valid = True
                
                # Check left part (0 to i-1)
                for k in range(i-1):
                    if nums[k] >= nums[k+1]:
                        is_valid = False
                        break
                
                if not is_valid:
                    continue
                
                # Check right part (j+1 to n-1)
                for k in range(j+1, n-1):
                    if nums[k] >= nums[k+1]:
                        is_valid = False
                        break
                
                if not is_valid:
                    continue
                
                # Check connection between left and right parts
                if i > 0 and j < n-1:
                    if nums[i-1] >= nums[j+1]:
                        is_valid = False
                
                if is_valid:
                    count += 1
        
        return count