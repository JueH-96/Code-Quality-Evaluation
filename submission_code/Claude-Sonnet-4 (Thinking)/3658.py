class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Handle case with no missing values
        if -1 not in nums:
            return max(abs(nums[i] - nums[i + 1]) for i in range(n - 1))
        
        # Handle case with all missing values
        if all(x == -1 for x in nums):
            return 0
        
        # Get all non-missing values
        known_values = [x for x in nums if x != -1]
        min_val, max_val = min(known_values), max(known_values)
        
        # Generate candidate values for x and y
        candidates = set(known_values)
        
        # Add values around the range of known values
        range_size = max_val - min_val
        for val in known_values:
            for delta in range(-range_size, range_size + 1):
                if val + delta > 0:
                    candidates.add(val + delta)
        
        candidates = sorted(candidates)
        
        def evaluate(x, y):
            missing_positions = [i for i in range(n) if nums[i] == -1]
            
            # For efficiency, limit brute force to small cases
            if len(missing_positions) > 18:
                return self.heuristic_solve(nums, x, y)
            
            best_max_diff = float('inf')
            
            # Try all possible assignments of x and y to missing positions
            for mask in range(2**len(missing_positions)):
                temp_nums = nums[:]
                for i, pos in enumerate(missing_positions):
                    temp_nums[pos] = x if (mask >> i) & 1 == 0 else y
                
                max_diff = max(abs(temp_nums[i] - temp_nums[i + 1]) for i in range(n - 1))
                best_max_diff = min(best_max_diff, max_diff)
            
            return best_max_diff
        
        result = float('inf')
        for x in candidates:
            for y in candidates:
                result = min(result, evaluate(x, y))
        
        return result
    
    def heuristic_solve(self, nums, x, y):
        # For large cases, use segment-based approach
        n = len(nums)
        temp_nums = nums[:]
        
        # Process segments of consecutive -1s
        i = 0
        while i < n:
            if temp_nums[i] == -1:
                start = i
                while i < n and temp_nums[i] == -1:
                    i += 1
                end = i - 1
                
                # Fill this segment optimally
                length = end - start + 1
                left_val = temp_nums[start - 1] if start > 0 else None
                right_val = temp_nums[end + 1] if end < n - 1 else None
                
                # Choose values to create smooth transition
                if left_val is None and right_val is None:
                    # Fill with same value
                    for j in range(start, end + 1):
                        temp_nums[j] = x
                elif left_val is None:
                    # Choose value closer to right_val
                    val = x if abs(x - right_val) <= abs(y - right_val) else y
                    for j in range(start, end + 1):
                        temp_nums[j] = val
                elif right_val is None:
                    # Choose value closer to left_val
                    val = x if abs(x - left_val) <= abs(y - left_val) else y
                    for j in range(start, end + 1):
                        temp_nums[j] = val
                else:
                    # Transition from left_val to right_val
                    # Use the value closer to left_val at start, closer to right_val at end
                    if length == 1:
                        # Choose value that minimizes max difference
                        diff_x = max(abs(x - left_val), abs(x - right_val))
                        diff_y = max(abs(y - left_val), abs(y - right_val))
                        temp_nums[start] = x if diff_x <= diff_y else y
                    else:
                        # Fill to create transition
                        for j in range(start, end + 1):
                            if j <= start + length // 2:
                                temp_nums[j] = x if abs(x - left_val) <= abs(y - left_val) else y
                            else:
                                temp_nums[j] = x if abs(x - right_val) <= abs(y - right_val) else y
            else:
                i += 1
        
        return max(abs(temp_nums[i] - temp_nums[i + 1]) for i in range(n - 1))