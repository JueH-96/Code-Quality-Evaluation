class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def can_mark_all_by_second(t):
            # Necessary condition: we need sum(nums) + n operations, and we have t seconds
            if sum(nums) + n > t:
                return False
            
            # For each index i, find all seconds <= t where changeIndices[s] == i
            mark_opportunities = [[] for _ in range(n + 1)]
            for s in range(1, t + 1):
                idx = changeIndices[s - 1]  # Convert to 0-indexed for changeIndices
                mark_opportunities[idx].append(s)
            
            # Check if all indices have at least one marking opportunity
            for i in range(1, n + 1):
                if not mark_opportunities[i]:
                    return False
            
            # Sort indices in decreasing order of nums[i] (priority to indices that need more decrements)
            indices_by_priority = sorted(range(1, n + 1), key=lambda i: nums[i - 1], reverse=True)
            
            mark_time = [0] * (n + 1)
            used_times = set()
            
            for idx in indices_by_priority:
                # Find the latest available marking time for idx
                available_times = [s for s in mark_opportunities[idx] if s not in used_times]
                if not available_times:
                    return False
                
                chosen_time = max(available_times)
                mark_time[idx] = chosen_time
                used_times.add(chosen_time)
            
            # Now, check if the assignment is feasible
            # Sort indices by their marking time
            indices_by_mark_time = sorted(range(1, n + 1), key=lambda i: mark_time[i])
            
            time_used_for_decrements = 0
            for j, idx in enumerate(indices_by_mark_time):
                # We need nums[idx-1] decrements for this index (nums is 0-indexed)
                decrements_needed = nums[idx - 1]
                time_used_for_decrements += decrements_needed
                
                # We mark idx at mark_time[idx]
                # Before this time, we have mark_time[idx] - 1 seconds
                # Out of these, j seconds are used for marking the first j indices
                # So, we have mark_time[idx] - 1 - j seconds available for decrementing
                available_time_for_decrements = mark_time[idx] - 1 - j
                if time_used_for_decrements > available_time_for_decrements:
                    return False
            
            return True
        
        # Binary search for the earliest second
        left, right = 1, m
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if can_mark_all_by_second(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result