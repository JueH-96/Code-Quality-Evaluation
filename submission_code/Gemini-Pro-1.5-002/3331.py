class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        nums.sort()
        n = len(nums)
        
        for i in range(n):
            if nums[i] < k:
                count +=1
            else:
                break
        return count