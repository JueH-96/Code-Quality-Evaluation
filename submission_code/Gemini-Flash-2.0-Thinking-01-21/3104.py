class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        counts = [0] * (n + 1) # size n+1 to accommodate values from 0 to n (though nums[i] < n) and k up to n
        for x in nums:
            counts[x] += 1
        prefix_counts = [0] * (n + 1)
        prefix_counts[0] = counts[0]
        for i in range(1, n + 1):
            prefix_counts[i] = prefix_counts[i-1] + counts[i]
        
        valid_ways_count = 0
        for k in range(n + 1):
            if counts[k] > 0:
                continue
            sum_less_than_k = 0
            if k > 0:
                sum_less_than_k = prefix_counts[k-1]
            if sum_less_than_k == k:
                valid_ways_count += 1
                
        return valid_ways_count