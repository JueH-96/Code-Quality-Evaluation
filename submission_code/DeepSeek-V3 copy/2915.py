from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # First, transform the nums array to a binary array where 1 indicates nums[i] % modulo == k
        binary = [1 if num % modulo == k else 0 for num in nums]
        
        # Now, we need to find the number of subarrays where the sum of the binary array modulo modulo equals k
        # We can use the prefix sum approach to solve this problem
        
        # Initialize the prefix sum dictionary with 0: 1 because a prefix sum of 0 has occurred once before the array starts
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1
        
        current_prefix = 0
        result = 0
        
        for num in binary:
            current_prefix += num
            # We need to find (current_prefix - target) % modulo == 0
            # Since target is k, we need (current_prefix - k) % modulo == 0
            # So, we look for (current_prefix - k) % modulo == 0 in the prefix_counts
            # Which is equivalent to current_prefix % modulo == k
            # Wait, no. Let's think differently.
            # We need (current_prefix - previous_prefix) % modulo == k
            # Which is equivalent to (current_prefix - k) % modulo == previous_prefix % modulo
            # So, we can rearrange to previous_prefix % modulo == (current_prefix - k) % modulo
            # So, we need to find the count of prefix_counts where prefix % modulo == (current_prefix - k) % modulo
            # So, we can compute (current_prefix - k) % modulo and look for it in prefix_counts
            target = (current_prefix - k) % modulo
            result += prefix_counts.get(target, 0)
            # Update the current_prefix % modulo in the prefix_counts
            prefix_counts[current_prefix % modulo] += 1
        
        return result