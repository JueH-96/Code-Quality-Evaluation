class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        or_sum = 0
        for num in nums:
            or_sum |= num
        
        max_or = 0
        for i in range(n):
            temp_or = or_sum
            for j in range(n):
                if i != j:
                    temp_or |= (nums[j] << k)
            max_or = max(max_or, temp_or)
            
        return max_or