from bisect import bisect_right
from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Convert changeIndices to 0-based index
        changeIndices = [x - 1 for x in changeIndices]
        
        # Preprocess occurrences for each index
        occurrences = [[] for _ in range(n)]
        for idx, val in enumerate(changeIndices):
            occurrences[val].append(idx)
        
        # Check if any index has no occurrences
        for i in range(n):
            if not occurrences[i]:
                return -1
        
        # Compute first_occurrence_1based for each index
        first_occurrence_1based = [occurrences[i][0] + 1 for i in range(n)]
        max_first_occurrence = max(first_occurrence_1based) if first_occurrence_1based else 0
        
        sum_nums = sum(nums)
        sum_nums_plus_n = sum_nums + n
        
        if sum_nums_plus_n > m:
            return -1
        
        low = max(sum_nums_plus_n, max_first_occurrence)
        high = m
        
        if low > high:
            return -1
        
        answer = -1
        
        while low <= high:
            mid = (low + high) // 2
            
            feasible = True
            s_list = []
            
            for i in range(n):
                # Find last occurrence of i in changeIndices[0..mid-1] (0-based)
                idx = bisect_right(occurrences[i], mid - 1) - 1
                if idx < 0:
                    feasible = False
                    break
                last_occurrence = occurrences[i][idx]
                s_i = last_occurrence + 1  # Convert to 1-based second
                s_list.append((s_i, i))
            
            if not feasible:
                low = mid + 1
                continue
            
            # Sort by s_i
            s_list.sort()
            
            feasible = True
            for k in range(n):
                s_i, i = s_list[k]
                available = (s_i - 1) - k
                if available < nums[i]:
                    feasible = False
                    break
            
            if feasible:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer