class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < k:
            return -1
        
        nums.sort(reverse=True)
        
        count = 0
        collected = 0
        
        for num in nums:
            if collected < k:
                collected += num
                count += 1
            else:
                break
        
        if collected < k:
            return -1
        
        return count