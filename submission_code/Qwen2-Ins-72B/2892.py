class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = max(nums)
        expected_length = max_num + 1
        if len(nums) != expected_length:
            return False
        
        num_counts = {}
        for num in nums:
            if num > max_num:
                return False
            if num in num_counts:
                num_counts[num] += 1
            else:
                num_counts[num] = 1
        
        for num in range(1, max_num):
            if num not in num_counts or num_counts[num] != 1:
                return False
        
        if max_num not in num_counts or num_counts[max_num] != 2:
            return False
        
        return True