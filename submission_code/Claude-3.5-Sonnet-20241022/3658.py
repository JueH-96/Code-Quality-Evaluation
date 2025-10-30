class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # If all elements are -1, we can make them all equal
        if all(x == -1 for x in nums):
            return 0
            
        # Get positions of missing elements
        missing_pos = [i for i, x in enumerate(nums) if x == -1]
        # Get actual numbers (non -1)
        actual = [x for x in nums if x != -1]
        
        if not actual:  # If all elements are missing
            return 0
            
        # Sort actual numbers
        actual.sort()
        
        result = float('inf')
        # Try all possible values between min and max of actual numbers
        min_val = min(actual)
        max_val = max(actual)
        
        # Helper function to check if a combination is valid
        def check(x: int, y: int) -> int:
            # Create new array with x and y replacements
            new_nums = nums.copy()
            # Try both orderings of x,y for the missing positions
            arrangements = [(x, y), (y, x)]
            min_diff = float('inf')
            
            for arr in arrangements:
                temp = new_nums.copy()
                for i, pos in enumerate(missing_pos):
                    temp[pos] = arr[i % 2]
                
                # Calculate maximum difference
                max_diff = 0
                for i in range(1, len(temp)):
                    max_diff = max(max_diff, abs(temp[i] - temp[i-1]))
                min_diff = min(min_diff, max_diff)
            
            return min_diff
        
        # Binary search approach for finding optimal values
        left, right = min_val, max_val
        while left <= right:
            mid = (left + right) // 2
            # Try values around mid
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x = mid + dx
                    y = mid + dy
                    if x > 0 and y > 0:
                        result = min(result, check(x, y))
            
            # Check if we should search in left or right half
            curr_diff = check(mid, mid)
            left_diff = check(mid-1, mid-1) if mid > 1 else float('inf')
            right_diff = check(mid+1, mid+1)
            
            if left_diff < curr_diff:
                right = mid - 1
            elif right_diff < curr_diff:
                left = mid + 1
            else:
                break
                
        return result