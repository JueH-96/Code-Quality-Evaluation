class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(1 << n):
            selected = []
            for j in range(n):
                if (i >> j) & 1:
                    selected.append(j)
            
            k=len(selected)
            happy = True
            for j in range(n):
                if j in selected:
                    if k <= nums[j]:
                        happy = False
                        break
                else:
                    if k >= nums[j]:
                        happy = False
                        break
            if happy:
                count += 1
        return count