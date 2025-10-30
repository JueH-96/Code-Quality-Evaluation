class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # Helper function to get the square-free part of a number
        def square_free_part(n):
            i = 2
            result = 1
            while i * i <= n:
                count = 0
                while n % i == 0:
                    count += 1
                    n //= i
                if count % 2 == 1:
                    result *= i
                i += 1
            if n > 1:  # If there's a remaining prime factor
                result *= n
            return result
        
        # Group indices based on their square-free part
        from collections import defaultdict
        part_groups = defaultdict(list)
        for i in range(1, len(nums) + 1):  # 1-indexed
            part = square_free_part(i)
            part_groups[part].append(i)
        
        # Find the maximum element-sum of a part group
        max_sum = 0
        for part, indices in part_groups.items():
            element_sum = sum(nums[i-1] for i in indices)  # Adjust for 0-indexing in Python
            max_sum = max(max_sum, element_sum)
        
        return max_sum