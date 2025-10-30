from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        freq = defaultdict(int)
        freq[0] = 1  # Initialize with prefix sum 0
        current_mod = 0
        result = 0
        
        for num in nums:
            if num % modulo == k:
                current_mod += 1
            current_mod %= modulo  # Ensure current_mod is within modulo
            
            target = (current_mod - k) % modulo
            result += freq.get(target, 0)
            
            # Update the frequency map with the current_mod
            freq[current_mod] += 1
        
        return result