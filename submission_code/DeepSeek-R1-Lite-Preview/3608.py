from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        
        # Count frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Total subsets where all elements are divisible by g
        total = [0] * (max_num + 1)
        for g in range(1, max_num + 1):
            count = 0
            for num in nums:
                if num % g == 0:
                    count += 1
            total[g] = (1 << count) - 1  # 2^count - 1
        
        # Compute cnt[g]: number of subsets with GCD exactly g
        cnt = [0] * (max_num + 1)
        for g in range(max_num, 0, -1):
            cnt[g] = total[g]
            multiple = 2 * g
            while multiple <= max_num:
                cnt[g] -= cnt[multiple]
                multiple += g
        
        # Calculate the number of pairs
        result = 0
        for g in range(1, max_num + 1):
            if cnt[g] < 2:
                continue
            result = (result + cnt[g] * (cnt[g] - 1) // 2) % MOD
        
        return result