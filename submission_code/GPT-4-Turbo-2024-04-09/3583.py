class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        from collections import Counter
        
        # Calculate all possible gcd pairs
        n = len(nums)
        gcd_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        # Sort the gcd pairs
        gcd_pairs.sort()
        
        # Answer each query
        answer = [gcd_pairs[q] for q in queries]
        return answer