class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        M = max(nums)
        left = 0
        count_M = 0
        total = 0
        for right in range(n):
            if nums[right] == M:
                count_M += 1
            while count_M >= k and left <= right:
                total += n - right
                if nums[left] == M:
                    count_M -= 1
                left += 1
        return total