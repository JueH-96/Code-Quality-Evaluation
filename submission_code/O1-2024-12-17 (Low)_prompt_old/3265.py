class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        import bisect
        
        n = len(nums)
        # Edge case: if no possible subarray of length >= 2 can satisfy |nums[i] - nums[j]| = k
        if n < 2:
            return 0
        
        # Prefix sums for fast subarray sum calculation: prefixSum[i] = sum(nums[0..i-1])
        prefixSum = [0] * (n + 1)
        for i in range(n):
            prefixSum[i+1] = prefixSum[i] + nums[i]
        
        # Dictionary to hold indices of each distinct value in nums
        from collections import defaultdict
        value_to_indices = defaultdict(list)
        for i, val in enumerate(nums):
            value_to_indices[val].append(i)
        
        # Sort the index lists for each value
        for val in value_to_indices:
            value_to_indices[val].sort()
        
        # We'll keep track of the best subarray sum found; if none found, return 0
        best_sum = None
        
        # Precompute a helper function that, given a sorted list of indices L,
        # returns an array M where M[i] = max(prefixSum[L[i] + 1], prefixSum[L[i+1] + 1], ..., prefixSum[L[end] + 1])
        def build_suffix_max(L):
            m = [0] * len(L)
            max_val = float('-inf')
            for i in range(len(L) - 1, -1, -1):
                idx = L[i]
                psum = prefixSum[idx + 1]  # sum up to and including idx
                if psum > max_val:
                    max_val = psum
                m[i] = max_val
            return m
        
        # For each value, check pairs with value+k and value-k
        for val, inds in value_to_indices.items():
            # For each target in [val+k, val-k]
            for target in (val + k, val - k):
                if target in value_to_indices:
                    target_inds = value_to_indices[target]
                    # Build suffix max array for the target indices
                    suffix_max = build_suffix_max(target_inds)
                    
                    # For each i in inds, we want subarray i..j with j > i and nums[j] = target
                    # We'll find the leftmost index in target_inds that is strictly > i
                    for i_idx in inds:
                        # We need j_idx > i_idx
                        pos = bisect.bisect_right(target_inds, i_idx)
                        if pos < len(target_inds):
                            # suffix_max[pos] = maximum prefixSum of subarray end among all target_inds >= pos
                            cur_sum = suffix_max[pos] - prefixSum[i_idx]
                            if best_sum is None or cur_sum > best_sum:
                                best_sum = cur_sum
        
        # If no valid subarray was found, best_sum would be None
        return best_sum if best_sum is not None else 0