from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        freq = defaultdict(int)
        freq[0] = 1  # Initialize with the sum 0 occurring once
        current_sum = 0
        result = 0
        
        for num in nums:
            # Update the current sum based on whether the current number meets the condition
            current_sum += 1 if (num % modulo == k) else 0
            rem = current_sum % modulo
            target = (rem - k) % modulo  # Calculate the target remainder
            
            # Add the frequency of the target remainder to the result
            result += freq.get(target, 0)
            
            # Update the frequency map with the current remainder
            freq[rem] += 1
        
        return result