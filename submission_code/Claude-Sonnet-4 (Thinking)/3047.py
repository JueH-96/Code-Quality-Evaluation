class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free_part(n):
            result = 1
            i = 2
            while i * i <= n:
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                if count % 2 == 1:
                    result *= i
                i += 1
            if n > 1:
                result *= n
            return result
        
        n = len(nums)
        groups = {}
        
        for i in range(1, n + 1):
            sf = square_free_part(i)
            if sf not in groups:
                groups[sf] = []
            groups[sf].append(i)
        
        max_sum = 0
        for indices in groups.values():
            current_sum = sum(nums[i - 1] for i in indices)  # Convert to 0-indexed
            max_sum = max(max_sum, current_sum)
        
        return max_sum