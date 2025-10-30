class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # prefix_sum_before is sum of nums up to but not including the current index
        prefix_sum = 0
        # For each value v, min_prefix[v] will store the minimum prefix_sum_before
        # seen so far at an index where nums[index] == v
        min_prefix = {}
        res = None  # Will hold the maximum sum of a good subarray found
        
        for x in nums:
            # Save the prefix sum before including x
            prefix_before = prefix_sum
            # Compute the prefix sum including x
            prefix_after = prefix_before + x
            
            # A good subarray ending here must start at some index s with nums[s] = x+k or x-k
            t1, t2 = x + k, x - k
            if t1 in min_prefix:
                candidate = prefix_after - min_prefix[t1]
                if res is None or candidate > res:
                    res = candidate
            if t2 in min_prefix:
                candidate = prefix_after - min_prefix[t2]
                if res is None or candidate > res:
                    res = candidate
            
            # Now consider this position as a potential start for future subarrays:
            # update the minimum prefix sum for value x
            if x not in min_prefix or prefix_before < min_prefix[x]:
                min_prefix[x] = prefix_before
            
            # Move forward in the prefix sum
            prefix_sum = prefix_after
        
        # If we never found a good subarray, return 0; otherwise return the best sum
        return res if res is not None else 0