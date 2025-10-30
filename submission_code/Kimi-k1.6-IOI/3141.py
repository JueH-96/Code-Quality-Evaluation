class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        def min_subarray_length(arr, target_sum):
            prefix_sum = 0
            prefix_map = {0: -1}
            min_len = float('inf')
            for i in range(len(arr)):
                prefix_sum += arr[i]
                if (prefix_sum - target_sum) in prefix_map:
                    current_len = i - prefix_map[prefix_sum - target_sum]
                    if current_len < min_len:
                        min_len = current_len
                if prefix_sum not in prefix_map:
                    prefix_map[prefix_sum] = i
            return min_len if min_len != float('inf') else -1
        
        n = len(nums)
        sum_nums = sum(nums)
        if sum_nums == 0:
            return -1
        
        k = target // sum_nums
        r = target % sum_nums
        
        if r == 0:
            return k * n
        else:
            extended = nums + nums
            min_len_r = min_subarray_length(extended, r)
            min_len_r_plus_s = min_subarray_length(extended, r + sum_nums)
            
            candidates = []
            if min_len_r != -1:
                candidates.append(k * n + min_len_r)
            if min_len_r_plus_s != -1 and k >= 1:
                candidates.append((k - 1) * n + min_len_r_plus_s)
            
            return min(candidates) if candidates else -1