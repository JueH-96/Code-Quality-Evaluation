class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            current_length = 1
            prev_parity = nums[l] % 2
            for r in range(l + 1, n):
                if nums[r] > threshold:
                    break
                curr_parity = nums[r] % 2
                if curr_parity == prev_parity:
                    break
                current_length += 1
                prev_parity = curr_parity
            max_length = max(max_length, current_length)
        return max_length