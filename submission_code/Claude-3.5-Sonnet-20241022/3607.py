class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def get_greatest_proper_divisor(n):
            # For 1, there is no proper divisor
            if n == 1:
                return 1
            
            # Find the greatest proper divisor
            for i in range(n // 2, 0, -1):
                if n % i == 0:
                    return i
            return 1
        
        def can_make_non_decreasing(num, prev, memo={}):
            # Base case: if number is less than previous, we can't proceed
            if num < prev:
                return float('inf')
            
            # If number is already greater or equal, no operations needed
            if num >= prev:
                return 0
            
            # Check memoized results
            key = (num, prev)
            if key in memo:
                return memo[key]
            
            # Get result and memoize
            memo[key] = float('inf')
            return memo[key]
        
        n = len(nums)
        operations = 0
        
        # Try to make array non-decreasing
        while True:
            is_non_decreasing = True
            made_change = False
            
            # Check if array is already non-decreasing
            for i in range(1, n):
                if nums[i] < nums[i-1]:
                    is_non_decreasing = False
                    break
            
            if is_non_decreasing:
                return operations
            
            # Try one round of operations
            for i in range(n-1):
                if nums[i] > nums[i+1]:
                    gpd = get_greatest_proper_divisor(nums[i])
                    if gpd == 1:  # Can't reduce further
                        return -1
                    nums[i] //= gpd
                    made_change = True
                    operations += 1
            
            # If we couldn't make any changes but array is still not non-decreasing
            if not made_change:
                return -1