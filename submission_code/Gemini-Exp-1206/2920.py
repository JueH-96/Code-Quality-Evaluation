class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if len(set(nums)) == 1:
            return 0
        
        pos = {}
        for i, num in enumerate(nums):
            if num not in pos:
                pos[num] = []
            pos[num].append(i)
        
        ans = float('inf')
        for num in pos:
            max_dist = 0
            for i in range(len(pos[num]) - 1):
                max_dist = max(max_dist, pos[num][i+1] - pos[num][i])
            max_dist = max(max_dist, pos[num][0] + n - pos[num][-1])
            ans = min(ans, max_dist // 2)
        
        return ans