class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                temp = nums[:i] + nums[j+1:]
                if all(temp[k] < temp[k+1] for k in range(len(temp)-1)):
                    count += 1

        return count