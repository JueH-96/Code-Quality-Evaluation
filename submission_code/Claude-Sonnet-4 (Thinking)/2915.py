class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        
        count = 0
        prefix_count = defaultdict(int)
        prefix_sum = 0
        
        # Initially, we have an empty prefix with sum 0
        prefix_count[0] = 1
        
        for num in nums:
            # Add 1 if num % modulo == k, else 0
            if num % modulo == k:
                prefix_sum += 1
            
            # For subarrays ending at current position, we need a prefix with sum s such that
            # (prefix_sum - s) % modulo == k
            # This means s % modulo should be (prefix_sum - k) % modulo
            target = (prefix_sum - k) % modulo
            count += prefix_count[target]
            
            # Add current prefix sum to the count
            prefix_count[prefix_sum % modulo] += 1
        
        return count