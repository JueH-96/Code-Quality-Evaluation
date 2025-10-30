class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            if nums[i + 1] == nums[i] + 1:  # Check for the start of an alternating subarray
                length = 2  # We have at least two elements
                for j in range(i + 2, n):
                    if (j - i) % 2 == 0:  # Even index in the subarray
                        if nums[j] == nums[j - 1] - 1:
                            length += 1
                        else:
                            break
                    else:  # Odd index in the subarray
                        if nums[j] == nums[j - 1] + 1:
                            length += 1
                        else:
                            break
                max_length = max(max_length, length)
        
        return max_length