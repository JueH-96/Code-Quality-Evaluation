class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        count = 0
        
        # Check k = 0 (select no students)
        if sorted_nums[0] > 0:
            count += 1
        
        # Check k = 1 to n-1
        for k in range(1, n):
            # For k selected students:
            # - The largest selected value must be < k
            # - The smallest non-selected value must be > k
            if sorted_nums[k-1] < k and sorted_nums[k] > k:
                count += 1
        
        # Check k = n (select all students)
        if sorted_nums[n-1] < n:
            count += 1
        
        return count