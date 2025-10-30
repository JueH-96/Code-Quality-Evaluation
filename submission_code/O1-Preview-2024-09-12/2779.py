class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0]*n
        count = 0
        ans = []
        for i, color in queries:
            old_color = nums[i]
            # Decrease count if old adjacent pairs existed
            if i > 0 and nums[i] != 0 and nums[i-1] == nums[i]:
                count -= 1
            if i < n - 1 and nums[i] != 0 and nums[i+1] == nums[i]:
                count -= 1
            nums[i] = color
            # Increase count if new adjacent pairs are formed
            if i > 0 and nums[i] != 0 and nums[i-1] == nums[i]:
                count += 1
            if i < n -1 and nums[i] != 0 and nums[i+1] == nums[i]:
                count += 1
            ans.append(count)
        return ans