class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import Counter
        
        def can_be_almost_equal(x, y):
            # Convert numbers to strings to compare their digit counts
            x_str = str(x)
            y_str = str(y)
            
            # If the lengths are different, they can't be made equal by swapping
            if len(x_str) != len(y_str):
                return False
            
            # Count the frequency of each digit
            x_count = Counter(x_str)
            y_count = Counter(y_str)
            
            # If the counts of digits are the same, they can be made equal by swapping
            return x_count == y_count
        
        count = 0
        n = len(nums)
        
        # Compare each pair (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                if can_be_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count