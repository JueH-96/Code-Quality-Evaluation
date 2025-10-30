class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        n = len(nums)
        gcd_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        gcd_pairs.sort()
        
        return [gcd_pairs[q] for q in queries]