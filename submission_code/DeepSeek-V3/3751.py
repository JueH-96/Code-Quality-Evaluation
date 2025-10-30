class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        total_k = nums.count(k)
        max_freq = total_k  # at least this, if we do nothing or choose a subarray of k's
        
        from collections import defaultdict
        # We need to process each a != k and find the longest contiguous a's
        current_val = None
        current_length = 0
        max_len_per_val = defaultdict(int)
        
        for num in nums:
            if num == current_val:
                current_length += 1
            else:
                if current_val is not None:
                    if current_val != k:
                        if current_length > max_len_per_val[current_val]:
                            max_len_per_val[current_val] = current_length
                current_val = num
                current_length = 1
        # After loop ends, process the last segment
        if current_val is not None and current_val != k:
            if current_length > max_len_per_val[current_val]:
                max_len_per_val[current_val] = current_length
        
        for a in max_len_per_val:
            candidate = max_len_per_val[a] + total_k
            if candidate > max_freq:
                max_freq = candidate
        
        return max_freq