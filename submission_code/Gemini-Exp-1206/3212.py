class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last_pos = {}
        for i, num in enumerate(nums):
            last_pos[num] = i
        
        ans = 1
        mod = 10**9 + 7
        
        end = 0
        count = 0
        for i, num in enumerate(nums):
            end = max(end, last_pos[num])
            if i == end:
                count += 1
        
        for _ in range(count - 1):
            ans = (ans * 2) % mod
        
        return ans