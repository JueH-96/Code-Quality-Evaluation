class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            # Check if the start of an alternating subarray is possible
            if nums[i + 1] == nums[i] + 1:
                current_length = 2
                # Check the rest of the subarray
                for j in range(i + 2, n):
                    if (j - i) % 2 == 0:
                        # Expect nums[j] to be nums[i]
                        if nums[j] != nums[i]:
                            break
                    else:
                        # Expect nums[j] to be nums[i] + 1
                        if nums[j] != nums[i] + 1:
                            break
                    current_length += 1
                max_length = max(max_length, current_length)
        
        return max_length