class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[:i] + nums[j+1:]
                if subarray == sorted(subarray) and len(set(subarray)) == len(subarray):
                    count += 1
                    
        return count