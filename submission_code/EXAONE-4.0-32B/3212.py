class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        last_occ = {}
        for i, num in enumerate(nums):
            last_occ[num] = i
        
        cur_max = -1
        count = 0
        for i in range(n):
            cur_max = max(cur_max, last_occ[nums[i]])
            if cur_max == i and i < n - 1:
                count += 1
        
        return pow(2, count, mod)