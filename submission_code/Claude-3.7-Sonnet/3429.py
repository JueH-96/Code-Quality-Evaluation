class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        for i in range(len(nums) - 1):
            # Check if adjacent elements have the same parity
            # (both even or both odd)
            if nums[i] % 2 == nums[i+1] % 2:
                return False
        # If we've checked all pairs and found no same-parity adjacents
        return True