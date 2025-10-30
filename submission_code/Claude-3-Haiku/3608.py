from collections import defaultdict
from math import gcd

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Count the frequency of each number in the array
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Calculate the number of pairs of subsequences with the same GCD
        total = 0
        for a in freq:
            for b in freq:
                if gcd(a, b) == 1:
                    total += freq[a] * freq[b]
        
        # Divide the total by 2 to avoid double counting
        return total // 2 % mod