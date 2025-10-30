class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # 1. Find the end index of the longest sequential prefix
        i = 0
        for j in range(1, len(nums)):
            if nums[j] == nums[j - 1] + 1:
                i = j
            else:
                break
        
        # 2. Calculate the sum of this prefix
        prefix_sum = sum(nums[:i + 1])
        
        # 3. Find the smallest integer not in nums that is >= prefix_sum
        existing = set(nums)
        candidate = prefix_sum
        while candidate in existing:
            candidate += 1
        
        return candidate