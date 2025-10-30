class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        expected_length = n + 1
        
        if len(nums) != expected_length:
            return False
        
        count = [0] * (n + 1)
        
        for num in nums:
            if num > n:
                return False
            count[num] += 1
        
        return count[n] == 2 and all(c == 1 for c in count[1:n])