class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        n = len(nums)
        gcd_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        gcd_pairs.sort()
        
        answer = []
        for q in queries:
            answer.append(gcd_pairs[q])
        
        return answer