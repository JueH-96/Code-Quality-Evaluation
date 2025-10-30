class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        original_bits = [bin(x).count('1') for x in nums]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if original_bits[i] != original_bits[j]:
                    if nums[i] > nums[j]:
                        return False
        return True