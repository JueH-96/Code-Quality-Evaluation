class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)
        
        if target > total:
            quotient, remainder = divmod(target, total)
            if remainder == 0:
                return quotient * n
            target = remainder
            min_len = quotient * n
        else:
            min_len = 0
        
        prefix_sum = 0
        sum_dict = {0: -1}
        curr_min = float('inf')
        
        for i in range(2 * n):
            prefix_sum += nums[i % n]
            if prefix_sum - target in sum_dict:
                curr_min = min(curr_min, i - sum_dict[prefix_sum - target])
            sum_dict[prefix_sum] = i
        
        if curr_min == float('inf'):
            return -1
        return curr_min + min_len