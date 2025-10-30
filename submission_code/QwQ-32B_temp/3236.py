class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # according to constraints, this case won't happen
        
        # Find the length of the longest sequential prefix
        length = 1
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                length += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        s = sum(nums[:length])
        
        # Find the smallest x >= s not present in nums
        num_set = set(nums)
        x = s
        while x in num_set:
            x += 1
        
        return x