class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            current_len = 1
            prev_parity = nums[l] % 2
            for i in range(l + 1, n):
                if nums[i] > threshold:
                    break
                current_parity = nums[i] % 2
                if current_parity != prev_parity:
                    current_len += 1
                    prev_parity = current_parity
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        return max_len