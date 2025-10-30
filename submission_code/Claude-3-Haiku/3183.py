class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k == n:
            return reduce(lambda x, y: x & y, nums)
        
        if k == 1:
            return reduce(lambda x, y: x | y, nums)
        
        result = 0
        for i in range(31):
            count = 0
            for num in nums:
                if num & (1 << i):
                    count += 1
            if count >= k:
                result |= (1 << i)
        
        return result