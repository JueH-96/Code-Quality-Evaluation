class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(math.gcd(nums[i], nums[j]))
        gcd_pairs.sort()
        ans = []
        for q in queries:
            ans.append(gcd_pairs[q])
        return ans