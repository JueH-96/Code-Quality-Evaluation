class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        for i in range(n - 1):
            if nums[i+1] - nums[i] != 1:
                continue
            current_length = 2
            current_direction = -1
            for j in range(i + 2, n):
                if nums[j] - nums[j-1] == current_direction:
                    current_length += 1
                    current_direction *= -1
                else:
                    break
            if current_length > max_length:
                max_length = current_length
        return max_length if max_length != -1 else -1