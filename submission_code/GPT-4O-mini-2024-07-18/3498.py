class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        half_n = n // 2
        
        # Count the frequency of each number in the first half and the corresponding numbers in the second half
        count = [0] * (k + 1)
        
        for i in range(half_n):
            count[nums[i]] += 1
            count[nums[n - i - 1]] += 1
        
        # Calculate the minimum changes needed
        min_changes = float('inf')
        
        for x in range(k + 1):
            changes = 0
            for j in range(k + 1):
                if abs(j - x) <= k:
                    changes += (half_n - count[j])
            min_changes = min(min_changes, changes)
        
        return min_changes