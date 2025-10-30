class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        
        count = 0
        prefix_count = 0
        freq = defaultdict(int)
        # The frequency of prefix_count == 0 is initially 1 (for subarrays starting at index 0)
        freq[0] = 1
        
        for num in nums:
            # Add 1 to prefix_count if current element % modulo == k, else 0
            prefix_count = (prefix_count + (1 if (num % modulo) == k else 0)) % modulo
            
            # To find the number of subarrays ending here which have sum % modulo = k:
            # We need prefix_count[l-1] = (prefix_count[r] - k) % modulo
            needed = (prefix_count - k) % modulo
            
            count += freq.get(needed, 0)
            
            # Update frequency of current prefix_count
            freq[prefix_count] += 1
        
        return count