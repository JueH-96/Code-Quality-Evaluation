class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        l, r = 0, len(nums) - k - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] != nums[mid + 1]:
                r = mid - 1
            else:
                l = mid + 1
        l, r = mid + 1, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] != nums[mid + 1]:
                l = mid + 1
            else:
                r = mid - 1
        return max(mid - l + 1, r - mid)