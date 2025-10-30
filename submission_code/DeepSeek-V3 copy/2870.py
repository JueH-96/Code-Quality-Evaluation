class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i+1] == nums[i] + 1:
                current_length = 2
                for j in range(i+2, n):
                    if (j - i) % 2 == 0:
                        if nums[j] == nums[i]:
                            current_length += 1
                        else:
                            break
                    else:
                        if nums[j] == nums[i+1]:
                            current_length += 1
                        else:
                            break
                if current_length > max_length:
                    max_length = current_length
        return max_length if max_length >= 2 else -1