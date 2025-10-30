class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changes = 0
        for x in range(k + 1):
            curr_changes = 0
            for i in range(n // 2):
                if abs(nums[i] - nums[n - 1 - i]) != x:
                    if nums[i] + x > k and nums[n-1-i] + x >k and abs(nums[i]-nums[n-1-i]) != x:
                        curr_changes = float('inf')
                        break
                    curr_changes +=1
            
            if x == 0:
                changes = curr_changes
            else:
                changes = min(changes, curr_changes)
        return changes