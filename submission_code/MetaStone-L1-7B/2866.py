class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 != 0:
                continue
            if nums[l] > threshold:
                continue
            current_len = 1
            prev_parity = nums[l] % 2
            for r in range(l + 1, n):
                current_val = nums[r]
                if current_val > threshold:
                    break
                current_parity = current_val % 2
                if current_parity != prev_parity:
                    current_len += 1
                    prev_parity = current_parity
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        return max_len