class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            current_length = 1
            max_length = max(max_length, current_length)
            previous_parity = 0  # since nums[l] is even
            for r in range(l + 1, n):
                if nums[r] > threshold:
                    break
                current_parity = nums[r] % 2
                if current_parity == previous_parity:
                    break
                current_length += 1
                max_length = max(max_length, current_length)
                previous_parity = current_parity
        return max_length