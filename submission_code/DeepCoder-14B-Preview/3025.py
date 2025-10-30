import sys
from collections import defaultdict

class Solution:
    def minOperations(self, nums: list, target: int) -> int:
        total = sum(nums)
        if total < target:
            return -1
        
        # Create a frequency dictionary where key is the exponent (k) such that element is 2^k
        freq = defaultdict(int)
        for num in nums:
            if num == 0:
                continue
            k = 0
            n = num
            while n > 1:
                n = n // 2
                k += 1
            freq[k] += 1
        
        max_bit = max(freq.keys()) if freq else 0
        operations = 0
        
        for b in reversed(range(0, 31)):  # Process each bit from 30 down to 0
            if (target >> b) & 1 == 0:
                continue  # Skip if this bit is not set in the target
            
            # Check if there are any elements at this bit or higher
            available = 0
            for m in range(b, max_bit + 1):
                if m in freq:
                    available += freq[m]
            if available == 0:
                return -1
            
            current = freq.get(b, 0)
            if current >= 1:
                continue  # Already have enough elements at this bit
            
            needed = 1 - current
            
            # Try to get the needed elements from higher bits
            for m in range(max_bit, b, -1):
                if m not in freq:
                    continue
                max_elements = freq[m] * (2 ** (m - b))
                if max_elements >= needed:
                    # Calculate how many elements we need to take from this m
                    elements_needed = (needed + (2 ** (m - b) - 1)) // (2 ** (m - b))
                    operations += elements_needed * (2 ** (m - b) - 1)
                    # Update the frequency dictionary
                    freq[m] -= elements_needed
                    if freq[m] == 0:
                        del freq[m]
                    freq[b] = freq.get(b, 0) + elements_needed * (2 ** (m - b))
                    needed -= elements_needed * (2 ** (m - b))
                    if needed <= 0:
                        break
                else:
                    # Take all elements from this m
                    operations += freq[m] * (2 ** (m - b) - 1)
                    freq[b] = freq.get(b, 0) + freq[m] * (2 ** (m - b))
                    needed -= freq[m] * (2 ** (m - b))
                    del freq[m]
            
            if needed > 0:
                return -1
        
        return operations