class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_prefix = {}
        ans = -10**18
        
        for j in range(n):
            num = nums[j]
            for cand in [num + k, num - k]:
                if cand in min_prefix:
                    s = prefix[j + 1] - min_prefix[cand]
                    if s > ans:
                        ans = s
            
            current_prefix_val = prefix[j]
            if num in min_prefix:
                if current_prefix_val < min_prefix[num]:
                    min_prefix[num] = current_prefix_val
            else:
                min_prefix[num] = current_prefix_val
        
        return ans if ans != -10**18 else 0