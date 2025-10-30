class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def get_square_free(n):
            """Get the square-free part of n"""
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
        
        # Group indices by their square-free parts
        groups = {}
        
        for i in range(1, len(nums) + 1):  # 1-indexed
            square_free = get_square_free(i)
            if square_free not in groups:
                groups[square_free] = 0
            groups[square_free] += nums[i - 1]  # nums is 0-indexed
        
        return max(groups.values())