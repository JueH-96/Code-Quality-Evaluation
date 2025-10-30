class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def can_mark_all(seconds):
            # Check if all indices 1..n appear in changeIndices[0:seconds]
            seen = set(changeIndices[:seconds])
            if len(seen) < n:
                return False
            
            # Find last occurrence of each index
            last_occurrence = {}
            for s in range(seconds):
                idx = changeIndices[s]
                last_occurrence[idx] = s
            
            # Plan: mark each index at its last occurrence
            # Create list of (second_to_mark_0indexed, index)
            mark_plan = []
            for idx in range(1, n + 1):
                mark_plan.append((last_occurrence[idx], idx))
            mark_plan.sort()
            
            # Simulate
            decrements_left = nums[:]  # 0-indexed copy
            seconds_used = 0
            
            for mark_at, idx in mark_plan:
                # We will mark index idx at second mark_at+1 (1-indexed)
                # We have seconds from seconds_used to mark_at-1 for decrements
                
                available = mark_at - seconds_used
                needed = decrements_left[idx - 1]
                
                if needed > available:
                    return False
                
                # Do the decrements for this index
                decrements_left[idx - 1] = 0
                available -= needed
                
                # Use leftover time on other indices
                for i in range(n):
                    if decrements_left[i] > 0 and available > 0:
                        dec = min(decrements_left[i], available)
                        decrements_left[i] -= dec
                        available -= dec
                
                # Update seconds_used to after the mark
                seconds_used = mark_at + 1
            
            # Check if all are zero
            return all(x == 0 for x in decrements_left)
        
        # Binary search
        left, right = 1, m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if can_mark_all(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result