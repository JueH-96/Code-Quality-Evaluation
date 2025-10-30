import math

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcdPairs = []
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                gcdPairs.append(math.gcd(nums[i], nums[j]))
        gcdPairs.sort()
        answer = []
        for q in queries:
            answer.append(gcdPairs[q])
        return answer