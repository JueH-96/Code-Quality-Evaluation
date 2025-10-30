class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        n = len(nums)
        bit_counts = [count_set_bits(num) for num in nums]
        
        for i in range(n):
            for j in range(n - 1):
                if nums[j] > nums[j + 1] and bit_counts[j] != bit_counts[j + 1]:
                    return False
                
                if nums[j] > nums[j + 1] and bit_counts[j] == bit_counts[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        
        return True