import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        
        def is_perfect_square(n):
            if n < 0:
                return False
            root = int(math.sqrt(n))
            return root * root == n

        groups = {}
        for i, num in enumerate(nums):
            temp = num
            factors = {}
            i = 2
            while i * i <= temp:
                count = 0
                while temp % i == 0:
                    temp //= i
                    count += 1
                if count % 2 != 0:
                    factors[i] = count
                i += 1
            if temp > 1:
                factors[temp] = 1
            
            key = tuple(sorted(factors.items()))
            if key not in groups:
                groups[key] = 0
            groups[key] += num

        return max(groups.values())