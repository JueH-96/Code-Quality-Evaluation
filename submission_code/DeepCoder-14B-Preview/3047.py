from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        groups = defaultdict(int)
        for num in nums:
            square_free = 1
            temp = num
            i = 2
            while i * i <= temp:
                if temp % i == 0:
                    exponent = 0
                    while temp % i == 0:
                        exponent += 1
                        temp //= i
                    if exponent % 2 != 0:
                        square_free *= i
                i += 1
            if temp > 1:
                square_free *= temp
            groups[square_free] += num
        if not groups:
            return 0
        return max(groups.values()) if groups else 0