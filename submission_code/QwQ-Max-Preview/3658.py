class Solution:
    def minDifference(self, nums: List[int]) -> int:
        non_missing = []
        leading_missing = 0
        i = 0
        n = len(nums)
        # Process leading missing elements
        while i < n and nums[i] == -1:
            leading_missing += 1
            i += 1
        
        gaps = []
        prev_val = None
        in_gap = False
        gap_length = 0
        
        while i < n:
            current_val = nums[i]
            if current_val == -1:
                if not in_gap and prev_val is not None:
                    in_gap = True
                    gap_start = i
                gap_length += 1
            else:
                if in_gap:
                    gaps.append((prev_val, current_val, gap_length))
                    in_gap = False
                    gap_length = 0
                non_missing.append(current_val)
                prev_val = current_val
            i += 1
        
        # Process trailing missing elements
        trailing_missing = gap_length if in_gap else 0
        
        # Handle case where all elements are missing
        if not non_missing:
            return 0
        
        max_d = 0
        
        for a, b, m in gaps:
            if a > b:
                a, b = b, a  # Ensure a <= b
            # Generate candidate x and y values
            candidates = [a, b, (a + b) // 2, (a + b + 1) // 2]
            # Ensure candidates are at least 1
            candidates = [max(1, c) for c in candidates]
            min_d = float('inf')
            # Iterate all possible pairs of candidates
            for x in candidates:
                for y in candidates:
                    d1 = abs(x - y)
                    d2 = min(abs(a - x), abs(a - y))
                    d3 = min(abs(x - b), abs(y - b))
                    current_d = max(d1, d2, d3)
                    if current_d < min_d:
                        min_d = current_d
            if min_d > max_d:
                max_d = min_d
        
        return max_d