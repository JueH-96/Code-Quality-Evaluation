from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # Transform the array to count the number of elements satisfying nums[i] % modulo == k
        transformed = [1 if num % modulo == k else 0 for num in nums]
        
        # Compute the prefix sum of the transformed array
        prefix = [0] * (len(transformed) + 1)
        for i in range(len(transformed)):
            prefix[i+1] = prefix[i] + transformed[i]
        
        # Use a dictionary to count the occurrences of each prefix sum modulo modulo
        count = defaultdict(int)
        result = 0
        
        for p in prefix:
            # We need to find (p - x) % modulo == k
            # Which is equivalent to x % modulo == (p - k) % modulo
            target = (p - k) % modulo
            result += count.get(target, 0)
            # Update the count for the current prefix sum modulo modulo
            count[p % modulo] += 1
        
        return result