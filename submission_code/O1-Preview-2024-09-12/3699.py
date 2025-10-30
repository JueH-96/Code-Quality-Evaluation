class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        from collections import defaultdict
        n = len(nums)
        ans = 0
        
        for q in range(1, n - 5):
            for r in range(q + 2, n - 3):
                counts_p = defaultdict(int)
                for p in range(0, q - 1):
                    if q - p > 1:
                        product = nums[p] * nums[r]
                        counts_p[product] += 1
                counts_s = defaultdict(int)
                for s in range(r + 2, n):
                    if s - r > 1:
                        product = nums[q] * nums[s]
                        counts_s[product] +=1
                common_products = set(counts_p.keys()) & set(counts_s.keys())
                for product in common_products:
                    ans += counts_p[product] * counts_s[product]
        return ans