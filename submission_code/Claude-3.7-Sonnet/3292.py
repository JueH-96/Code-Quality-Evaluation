class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        
        # Convert to 0-indexed
        changeIndices = [idx - 1 for idx in changeIndices]
        
        def is_possible(end_time):
            # Find the last occurrence of each index
            last_occurrence = {}
            for s in range(end_time):
                idx = changeIndices[s]
                last_occurrence[idx] = s
            
            # Check if all indices appear in changeIndices
            if len(last_occurrence) < n:
                return False
            
            # Sort indices by their last occurrence (to mark indices in optimal order)
            indices = sorted(range(n), key=lambda i: last_occurrence.get(i, float('inf')))
            
            # Simulate the process
            operations_used = 0
            
            for idx in indices:
                # Check if index appears in changeIndices
                if idx not in last_occurrence:
                    return False
                
                mark_time = last_occurrence[idx]
                decrements_needed = nums[idx]
                
                # Check if we have enough time to decrement before marking
                if operations_used + decrements_needed > mark_time:
                    return False
                
                # Decrement and mark
                operations_used += decrements_needed + 1  # +1 for marking
            
            return operations_used <= end_time
        
        # Binary search for earliest valid second
        left, right = 1, m
        while left < right:
            mid = (left + right) // 2
            if is_possible(mid):
                right = mid
            else:
                left = mid + 1
        
        if left <= m and is_possible(left):
            return left
        return -1