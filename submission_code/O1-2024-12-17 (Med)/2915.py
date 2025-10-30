class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        
        prefix_count = defaultdict(int)
        prefix_count[0] = 1  # Prefix sum 0 appears once initially
        ans = 0
        current_prefix = 0
        
        for num in nums:
            # Check if current element % modulo equals k
            x = 1 if (num % modulo) == k else 0
            
            # Update prefix sum mod 'modulo'
            current_prefix = (current_prefix + x) % modulo
            
            # We want current_prefix - k (mod modulo) to have appeared before
            needed = (current_prefix - k) % modulo
            ans += prefix_count.get(needed, 0)
            
            # Record the current prefix sum occurrence
            prefix_count[current_prefix] += 1
        
        return ans