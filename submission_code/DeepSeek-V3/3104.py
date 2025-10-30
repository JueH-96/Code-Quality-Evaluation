class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ways = 0
        
        # Check for selecting 0 students
        if all(nums[i] > 0 for i in range(n)):
            ways += 1
        
        # Check for selecting all students
        if all(nums[i] < n for i in range(n)):
            ways += 1
        
        # Check for selecting k students where 1 <= k < n
        for k in range(1, n):
            # Check if all selected students have nums[i] < k
            # and all unselected students have nums[i] > k
            # Since nums is sorted, the first k students are selected
            # So, nums[k-1] < k and nums[k] > k
            if nums[k-1] < k and nums[k] > k:
                ways += 1
        
        return ways