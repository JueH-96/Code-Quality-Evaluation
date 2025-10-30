class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        def canMarkAll(maxTime):
            # Find the last occurrence of each index within maxTime
            lastOccurrence = {}
            for s in range(maxTime):
                idx = changeIndices[s] - 1  # Convert to 0-indexed
                lastOccurrence[idx] = s
            
            # Check if all indices appear in changeIndices within maxTime
            for i in range(n):
                if i not in lastOccurrence:
                    return False
            
            # Calculate required decrements for each index
            requiredDecrements = nums[:]
            
            # Reserve seconds for marking (using last occurrences)
            reservedSeconds = set(lastOccurrence.values())
            
            # Count available seconds for decrementing
            availableSeconds = maxTime - len(reservedSeconds)
            
            # Check if we have enough seconds to decrement all values
            totalDecrements = sum(requiredDecrements)
            
            if totalDecrements <= availableSeconds:
                return True
            
            # If not enough seconds, we need to be more strategic
            # We need to ensure each index can be decremented to 0 before its last marking opportunity
            events = []
            for i in range(n):
                if nums[i] > 0:
                    events.append((lastOccurrence[i], nums[i]))  # (marking_time, decrements_needed)
            
            events.sort()  # Sort by marking time
            
            currentTime = 0
            totalDecrementsNeeded = 0
            
            for markingTime, decrementsNeeded in events:
                # Time available from currentTime to markingTime (exclusive)
                availableTimeForThis = markingTime - currentTime
                totalDecrementsNeeded += decrementsNeeded
                
                if totalDecrementsNeeded > availableTimeForThis:
                    return False
                
                currentTime = markingTime + 1  # Move past the marking time
            
            return True
        
        # Binary search on the answer
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