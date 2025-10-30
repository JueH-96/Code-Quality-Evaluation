class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        last_occurrence = {}
        for i in range(n):
            last_occurrence[nums[i]] = i
        
        rightmost = 0
        partitions = 0
        for i in range(n):
            rightmost = max(rightmost, last_occurrence[nums[i]])
            if i == rightmost:
                partitions += 1
        
        if partitions == 0:
            return 0
        
        MOD = 10**9 + 7
        result = pow(2, partitions - 1, MOD)
        return result