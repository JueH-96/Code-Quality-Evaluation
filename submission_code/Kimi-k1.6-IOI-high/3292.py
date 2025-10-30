class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if all elements from 1 to n are present in changeIndices
        required = set(changeIndices)
        for i in range(1, n + 1):
            if i not in required:
                return -1
        
        left, right = 1, m
        answer = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Compute last occurrence for each element in the first 'mid' elements
            last_occurrence = [0] * (n + 1)  # 1-based indexing
            feasible = True
            for s in range(mid):
                idx = changeIndices[s]
                last_occurrence[idx] = s + 1  # seconds are 1-indexed
            
            # Check if all elements have at least one occurrence in the first 'mid' seconds
            for i in range(1, n + 1):
                if last_occurrence[i] == 0:
                    feasible = False
                    break
            if not feasible:
                left = mid + 1
                continue
            
            # Check sum(nums) + n <= mid
            total_required = sum(nums) + n
            if total_required > mid:
                feasible = False
            else:
                # Check sum of last_occurrence >= total_required
                sum_last = sum(last_occurrence[1:n+1])
                if sum_last < total_required:
                    feasible = False
            
            if feasible:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer