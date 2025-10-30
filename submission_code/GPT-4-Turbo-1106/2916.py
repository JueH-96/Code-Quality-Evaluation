class Solution:
    def canSplitArray(self, nums: list[int], m: int) -> bool:
        def can_split(start, end, subarrays):
            if subarrays == 1:
                return sum(nums[start:end]) >= m
            for i in range(start + 1, end):
                if sum(nums[start:i]) >= m and can_split(i, end, subarrays - 1):
                    return True
            return False
        
        return can_split(0, len(nums), len(nums))

# Example usage:
# sol = Solution()
# print(sol.canSplitArray([2, 2, 1], 4))  # Output: true
# print(sol.canSplitArray([2, 1, 3], 5))  # Output: false
# print(sol.canSplitArray([2, 3, 3, 2, 3], 6))  # Output: true