import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums or len(nums) < 2:
            return []
        
        max_num = max(nums)
        n = len(nums)
        
        # Step 1: Compute frequency array
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        # Step 2: Compute count array
        count = [0] * (max_num + 2)  # +2 to avoid index issues
        for g in range(max_num, 0, -1):
            total_multiples = 0
            multiple = g
            while multiple <= max_num:
                total_multiples += freq[multiple]
                multiple += g
            pairs = total_multiples * (total_multiples - 1) // 2
            multiple = 2 * g
            while multiple <= max_num:
                pairs -= count[multiple]
                multiple += g
            count[g] = pairs
        
        # Step 3: Collect sorted_gcd
        sorted_gcd = []
        for g in range(1, max_num + 1):
            if count[g] > 0:
                sorted_gcd.append((g, count[g]))
        sorted_gcd.sort()  # Sort by GCD value in ascending order
        
        # Step 4: Compute prefix array
        prefix = [0]
        current = 0
        for g, c in sorted_gcd:
            current += c
            prefix.append(current)
        
        # Step 5: Process queries
        answer = []
        for k in queries:
            i = bisect.bisect_right(prefix, k)
            answer.append(sorted_gcd[i - 1][0])
        
        return answer