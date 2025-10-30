class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element
        x = self.find_dominant(nums)
        # Count its total frequency
        freq_total = self.count_frequency(nums, x)
        n = len(nums)
        # Iterate to find the split index
        count_left = 0
        for i in range(n-1):
            if nums[i] == x:
                count_left += 1
            freq_right = freq_total - count_left
            if count_left > (i + 1) / 2 and freq_right > (n - 1 - i) / 2:
                return i
        return -1

    def find_dominant(self, nums):
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        return candidate

    def count_frequency(self, nums, x):
        count = 0
        for num in nums:
            if num == x:
                count += 1
        return count