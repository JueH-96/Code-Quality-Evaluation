import bisect
from collections import defaultdict

class Solution:
    def minSizeSubarray(self, nums: list[int], target: int) -> int:
        sum_total = sum(nums)
        if sum_total == 0:
            return -1 if target != 0 else len(nums)
        
        concatenated = nums + nums
        n = len(concatenated)
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + concatenated[i]
        
        sum_indices = defaultdict(list)
        sum_indices[0].append(0)
        min_len = float('inf')
        
        for i in range(1, n + 1):
            current_sum = prefix_sum[i]
            target_sum = current_sum - target
            
            if target_sum in sum_indices:
                idx_list = sum_indices[target_sum]
                pos = bisect.bisect_left(idx_list, i) - 1
                if pos >= 0:
                    j = idx_list[pos]
                    current_len = i - j
                    if current_len < min_len:
                        min_len = current_len
            
            sum_indices[current_sum].append(i)
        
        if min_len != float('inf'):
            return min_len
        else:
            if target % sum_total != 0:
                return -1
            else:
                return (target // sum_total) * len(nums)