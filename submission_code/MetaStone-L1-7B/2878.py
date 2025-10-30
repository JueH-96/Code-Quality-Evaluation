class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n == 0:
            return True
        
        # Compute the difference array d
        d = [0] * n
        d[0] = nums[0]
        for i in range(1, n):
            d[i] = nums[i] - nums[i-1]
        
        # Compute the count array
        count = [0] * n
        count[0] = d[0]
        for i in range(1, n):
            if i < k:
                count[i] = d[i]
            else:
                count[i] = d[i] + count[i - k]
        
        # Check for i >= (n - k + 1), count[i] must be 0
        for i in range(n - k + 1, n):
            if count[i] != 0:
                return False
        
        # Check all count[i] >= 0
        for num in count:
            if num < 0:
                return False
        
        return True