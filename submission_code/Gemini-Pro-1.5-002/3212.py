class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        dp = [0] * (1 << n)
        dp[0] = 1
        for mask in range(1 << n):
            for i in range(n):
                if (mask >> i) & 1:
                    sub_mask = mask ^ (1 << i)
                    sub_nums = []
                    for j in range(n):
                        if (sub_mask >> j) & 1:
                            sub_nums.append(nums[j])
                    
                    valid = True
                    
                    
                    curr_nums = []
                    for j in range(i + 1):
                        if (mask >> j) & 1:
                            curr_nums.append(nums[j])
                    
                    
                    
                    
                    sets = set()
                    temp = []
                    for k in range(len(sub_nums)):
                        temp.append(sub_nums[k])
                        
                        
                    
                    sets.add(tuple(temp))
                    
                    temp = []
                    for k in range(len(curr_nums)):
                        temp.append(curr_nums[k])
                    
                    if tuple(temp) in sets:
                        valid = False
                    
                    if valid:
                        dp[mask] = (dp[mask] + dp[sub_mask]) % mod
        return dp[(1 << n) - 1]