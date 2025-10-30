class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ops = 0
        flipped = [0] * n
        
        for i in range(n):
            if nums[i] == 1:
                if flipped[i] % 2 == 1:
                    nums[i] = 0
            else:
                if flipped[i] % 2 == 0:
                    nums[i] = 1
        
        for i in range(n - 2):
            if nums[i] == 0:
                ops += 1
                flipped[i] += 1
                if i + 1 < n:
                    flipped[i+1] += 1
                if i + 2 < n:
                    flipped[i+2] += 1
        
        for i in range(n):
            if nums[i] == 1:
                if flipped[i] % 2 == 1:
                    nums[i] = 0
            else:
                if flipped[i] % 2 == 0:
                    nums[i] = 1
        
        if nums[n-1] == 0 or nums[n-2] == 0:
            return -1
        
        return ops