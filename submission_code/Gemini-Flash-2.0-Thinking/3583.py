import math

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcd_pairs = []
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                gcd_val = math.gcd(nums[i], nums[j])
                gcd_pairs.append(gcd_val)

        gcd_pairs.sort()

        results = []
        for query in queries:
            results.append(gcd_pairs[query])

        return results