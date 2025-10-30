class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if all indices appear in changeIndices
        indices_set = set(changeIndices)
        for i in range(1, n + 1):
            if i not in indices_set:
                return -1
        
        def canMarkAll(seconds):
            # Find last occurrence of each index within first 'seconds' seconds
            last_occurrence = {}
            for s in range(seconds):
                idx = changeIndices[s]
                last_occurrence[idx] = s + 1  # 1-indexed
            
            # Check if all indices have at least one occurrence
            if len(last_occurrence) < n:
                return False
            
            # Calculate total operations needed
            total_decrements = sum(nums)
            total_marks = n
            total_operations_needed = total_decrements + total_marks
            
            # Check if we have enough seconds
            if total_operations_needed > seconds:
                return False
            
            # Create events for when we must mark each index
            events = []
            for idx in range(1, n + 1):
                if idx in last_occurrence:
                    events.append((last_occurrence[idx], idx))
            events.sort()
            
            # Simulate the process
            operations_used = 0
            marked = set()
            
            for second, idx in events:
                # We need nums[idx-1] decrements before this second
                decrements_needed = nums[idx - 1]
                operations_used += decrements_needed
                
                # Check if we have enough operations up to this second
                if operations_used + len(marked) + 1 > second:
                    return False
                
                # Mark this index
                operations_used += 1
                marked.add(idx)
            
            return True
        
        # Binary search for the answer
        left, right = 1, m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMarkAll(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result