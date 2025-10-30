class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        S = sum(nums)
        doubled = nums + nums
        
        # Function to find minimal subarray with exact sum
        def min_subarray_sum(arr, t):
            prefix_map = {0: -1}
            current_sum = 0
            min_len = float('inf')
            for i, num in enumerate(arr):
                current_sum += num
                if (current_sum - t) in prefix_map:
                    start = prefix_map[current_sum - t]
                    min_len = min(min_len, i - start)
                if current_sum not in prefix_map:
                    prefix_map[current_sum] = i
            return min_len if min_len != float('inf') else -1
        
        # Check the doubled array for the target
        min_len = min_subarray_sum(doubled, target)
        
        if S == 0:
            return -1
        
        if target >= S:
            k = target // S
            for delta in [-1, 0, 1]:
                current_k = k + delta
                if current_k < 0:
                    continue
                rem = target - current_k * S
                if rem < 0:
                    continue
                if rem == 0:
                    candidate = current_k * n
                    if min_len == -1:
                        min_len = candidate
                    else:
                        min_len = min(min_len, candidate)
                else:
                    rem_len = min_subarray_sum(doubled, rem)
                    if rem_len != -1:
                        candidate = current_k * n + rem_len
                        if min_len == -1:
                            min_len = candidate
                        else:
                            min_len = min(min_len, candidate)
        
        return min_len if min_len != float('inf') else -1