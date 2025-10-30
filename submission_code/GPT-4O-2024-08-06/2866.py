class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        
        for start in range(n):
            if nums[start] % 2 == 0 and nums[start] <= threshold:
                current_length = 1
                last_parity = nums[start] % 2
                for end in range(start + 1, n):
                    if nums[end] <= threshold and nums[end] % 2 != last_parity:
                        current_length += 1
                        last_parity = nums[end] % 2
                    else:
                        break
                max_length = max(max_length, current_length)
        
        return max_length