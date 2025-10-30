class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            if nums[i + 1] == nums[i] + 1:
                length = 2
                for j in range(i + 2, n):
                    if (j - i) % 2 == 0:
                        if nums[j] != nums[i]:
                            break
                    else:
                        if nums[j] != nums[i] + 1:
                            break
                    length += 1
                max_length = max(max_length, length)
        
        return max_length