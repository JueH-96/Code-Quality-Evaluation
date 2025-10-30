from collections import defaultdict

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Create a frequency map for the possible values after applying the operation
        freq = defaultdict(int)
        for num in nums:
            # The possible range for each num is [num - k, num + k]
            # We can represent this as a range and count the frequency of each possible value
            # To optimize, we can use a difference array or a line sweep approach
            # Here, we'll use a simple approach by incrementing the start and decrementing the end+1
            start = num - k
            end = num + k
            freq[start] += 1
            freq[end + 1] -= 1
        
        # Now, we need to find the maximum frequency by processing the frequency map
        sorted_keys = sorted(freq.keys())
        max_beauty = 0
        current = 0
        for key in sorted_keys:
            current += freq[key]
            if current > max_beauty:
                max_beauty = current
        
        return max_beauty