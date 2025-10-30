class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the longest suffix with distinct elements
        seen = set()
        distinct_start = n  # Start with assuming we need to remove everything
        
        # Check from right to left
        for i in range(n - 1, -1, -1):
            if nums[i] in seen:
                # Found a duplicate, so distinct suffix starts after this position
                break
            seen.add(nums[i])
            distinct_start = i
        
        # If distinct_start is 0, the whole array is already distinct
        if distinct_start == 0:
            return 0
        
        # Calculate number of operations needed to remove elements before distinct_start
        # Each operation removes 3 elements, so we need ceil(distinct_start / 3) operations
        return (distinct_start + 2) // 3