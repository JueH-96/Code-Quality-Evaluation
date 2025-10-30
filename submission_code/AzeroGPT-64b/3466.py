class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        result = 0
        current_and = 0
        left = 0

        for right in range(len(nums)):
            current_and &= nums[right]

            while current_and != k:
                current_and &= ~nums[left]
                left += 1

            result += left + 1

        return result