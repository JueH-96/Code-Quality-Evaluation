class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def canMarkAll(seconds: int) -> bool:
            # Convert to 0-based indexing
            indices = [x - 1 for x in changeIndices[:seconds]]
            
            # Get last occurrence of each index
            last_occurrence = {}
            for i in range(seconds):
                last_occurrence[indices[i]] = i
                
            # Check if all indices appear
            if len(last_occurrence) < n:
                return False
                
            decrements_needed = sum(nums)
            marked = [False] * n
            available_decrements = 0
            
            # Work backwards from the end
            for i in range(seconds-1, -1, -1):
                curr_idx = indices[i]
                
                # If this is the last occurrence of current index
                if i == last_occurrence[curr_idx]:
                    if nums[curr_idx] > available_decrements:
                        return False
                    available_decrements -= nums[curr_idx]
                    marked[curr_idx] = True
                else:
                    available_decrements += 1
                    
            return all(marked)
            
        # Binary search on the answer
        left, right = 1, m
        ans = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canMarkAll(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
                
        return ans