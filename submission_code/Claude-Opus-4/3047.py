class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def get_square_free(n):
            # Get the square-free part of n
            # This is the product of all prime factors that appear an odd number of times
            result = 1
            d = 2
            while d * d <= n:
                count = 0
                while n % d == 0:
                    n //= d
                    count += 1
                if count % 2 == 1:
                    result *= d
                d += 1
            if n > 1:
                result *= n
            return result
        
        n = len(nums)
        # Group indices by their square-free part
        groups = {}
        
        for i in range(1, n + 1):
            sf = get_square_free(i)
            if sf not in groups:
                groups[sf] = []
            groups[sf].append(i)
        
        max_sum = 0
        
        # For each group, calculate the sum of elements at those indices
        for indices in groups.values():
            current_sum = sum(nums[idx - 1] for idx in indices)
            max_sum = max(max_sum, current_sum)
        
        return max_sum