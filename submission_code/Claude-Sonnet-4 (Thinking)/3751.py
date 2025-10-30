class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        counter = Counter(nums)
        total_freq_k = counter[k]
        max_freq = total_freq_k  # Case of not modifying anything
        
        # Try all possible values that appear in nums
        for v in counter:
            if v == k:
                continue  # Adding 0 doesn't change anything
            
            # Create the contribution array
            # +1 for elements that become k, -1 for elements that were k but no longer are
            contrib = [1 if num == v else (-1 if num == k else 0) for num in nums]
            
            # Find maximum subarray sum using Kadane's algorithm
            max_subarray_sum = 0
            current_sum = 0
            for c in contrib:
                current_sum = max(c, current_sum + c)
                max_subarray_sum = max(max_subarray_sum, current_sum)
            
            max_freq = max(max_freq, total_freq_k + max_subarray_sum)
        
        return max_freq