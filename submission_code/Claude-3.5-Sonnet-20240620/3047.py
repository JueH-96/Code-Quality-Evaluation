class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Function to check if a number is a perfect square
        def is_perfect_square(num):
            return int(num**0.5)**2 == num
        
        # Group numbers by their prime factors
        groups = {}
        for i, num in enumerate(nums):
            key = 1
            for factor in range(2, int(num**0.5) + 1):
                count = 0
                while num % factor == 0:
                    num //= factor
                    count += 1
                if count % 2:
                    key *= factor
            if num > 1:
                key *= num
            groups.setdefault(key, []).append(nums[i])
        
        # Find the maximum sum for each group
        max_sum = max(nums)
        for group in groups.values():
            group.sort(reverse=True)
            group_sum = sum(group[:2])
            max_sum = max(max_sum, group_sum)
        
        return max_sum