class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        from collections import defaultdict
        
        max_val = max(nums)
        sieve = [0] * (max_val + 1)
        for i in range(1, max_val + 1):
            for j in range(i, max_val + 1, i):
                sieve[j] += 1
        
        gcd_count = defaultdict(int)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                gcd_val = gcd(nums[i], nums[j])
                gcd_count[gcd_val] += 1
        
        gcd_pairs = sorted(gcd_count.keys())
        prefix_sum = [0] * (len(gcd_pairs) + 1)
        for i in range(len(gcd_pairs)):
            prefix_sum[i + 1] = prefix_sum[i] + gcd_count[gcd_pairs[i]]
        
        result = []
        for query in queries:
            idx = bisect.bisect_left(prefix_sum, query + 1)
            result.append(gcd_pairs[idx - 1])
        
        return result