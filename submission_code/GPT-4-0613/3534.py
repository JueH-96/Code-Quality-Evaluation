class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import defaultdict
        count = defaultdict(int)
        for num in nums:
            digits = [0]*10
            while num:
                digits[num%10] += 1
                num //= 10
            for i, d in enumerate(digits):
                if d:
                    count[i] += 1
        return sum(c*(c-1)//2 for c in count.values())