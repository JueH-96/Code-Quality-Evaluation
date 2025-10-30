class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_length = 1
                max_length = max(max_length, current_length)
                prev_parity = nums[l] % 2
                for r in range(l + 1, n):
                    if nums[r] > threshold:
                        break
                    current_parity = nums[r] % 2
                    if prev_parity == current_parity:
                        break
                    current_length += 1
                    max_length = max(max_length, current_length)
                    prev_parity = current_parity
        return max_length