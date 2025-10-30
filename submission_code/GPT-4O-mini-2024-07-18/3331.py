class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        operations = 0
        
        for num in nums:
            if num < k:
                operations += 1
            else:
                break
        
        return operations