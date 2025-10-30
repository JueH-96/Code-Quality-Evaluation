class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Step 1: Identify the longest sequential prefix
        # Start from the first element and keep going while nums[i] == nums[i-1] + 1
        prefix_length = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                prefix_length += 1
            else:
                break
        
        # Step 2: Sum the elements of this longest sequential prefix
        prefix_sum = sum(nums[:prefix_length])
        
        # Step 3: Find the smallest missing integer x >= prefix_sum
        existing = set(nums)  # for O(1) lookup
        candidate = prefix_sum
        while True:
            if candidate not in existing:
                return candidate
            candidate += 1