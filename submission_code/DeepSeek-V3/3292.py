class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Precompute the last occurrence of each index in changeIndices
        last_occurrence = {}
        for s in range(m):
            idx = changeIndices[s]
            last_occurrence[idx] = s + 1  # 1-based seconds
        
        # Check if all indices have at least one occurrence in changeIndices
        for i in range(1, n+1):
            if i not in last_occurrence:
                return -1
        
        # Binary search for the earliest second
        left = 1
        right = m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            # Check if it's possible to mark all indices by mid seconds
            possible = True
            # Calculate the required time for each index
            total_time = 0
            for i in range(1, n+1):
                # The last occurrence of i must be <= mid
                if last_occurrence[i] > mid:
                    possible = False
                    break
                # Time needed to reduce nums[i-1] to 0
                # Each decrement takes 1 second
                # We need to perform nums[i-1] decrements
                # But we can only perform decrements before the last occurrence
                # So the total time is nums[i-1] + 1 (for marking)
                # But since we can perform decrements in any order, we need to ensure that the total time is <= mid
                # So the total time is nums[i-1] + 1 <= mid
                if nums[i-1] + 1 > mid:
                    possible = False
                    break
                total_time += nums[i-1] + 1
            if possible and total_time <= mid:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result