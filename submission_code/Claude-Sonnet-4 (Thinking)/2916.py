class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 2:
            return True
        
        # Precompute prefix sums for efficient range sum queries
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        def get_sum(start, end):
            return prefix_sum[end + 1] - prefix_sum[start]
        
        memo = {}
        
        def can_split(start, end):
            # Base cases
            if start == end:
                return True  # Single element
            if end - start + 1 == 2:
                return True  # Two elements can always be split into two single elements
            
            if (start, end) in memo:
                return memo[(start, end)]
            
            # Try all possible splits
            for i in range(start, end):
                left_sum = get_sum(start, i)
                right_sum = get_sum(i + 1, end)
                left_len = i - start + 1
                right_len = end - i
                
                # Check if this split is valid
                left_valid = (left_len == 1) or (left_sum >= m)
                right_valid = (right_len == 1) or (right_sum >= m)
                
                if left_valid and right_valid:
                    # If both parts are valid, check if they can be further split
                    if can_split(start, i) and can_split(i + 1, end):
                        memo[(start, end)] = True
                        return True
            
            memo[(start, end)] = False
            return False
        
        return can_split(0, n - 1)