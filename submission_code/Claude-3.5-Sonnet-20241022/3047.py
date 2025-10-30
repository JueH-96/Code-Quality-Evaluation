class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Helper function to check if number is perfect square
        def is_perfect_square(x):
            root = int(x ** 0.5)
            return root * root == x
        
        # Helper function to check if product of two numbers is perfect square
        def is_pair_valid(x, y):
            return is_perfect_square(x * y)
        
        # Group numbers by their prime factorization pattern
        def get_pattern(num):
            pattern = []
            i = 2
            while i * i <= num:
                count = 0
                while num % i == 0:
                    count += 1
                    num //= i
                if count % 2 == 1:
                    pattern.append(i)
                i += 1
            if num > 1:
                pattern.append(num)
            return tuple(pattern)
        
        # Group numbers by their patterns
        groups = {}
        for i, num in enumerate(nums):
            pattern = get_pattern(num)
            if pattern not in groups:
                groups[pattern] = []
            groups[pattern].append(i)
            
        # Find maximum sum among compatible groups
        max_sum = max(nums)  # Initialize with max single element
        
        # Check within each group
        for indices in groups.values():
            curr_sum = sum(nums[i] for i in indices)
            max_sum = max(max_sum, curr_sum)
            
        # Check pairs of groups that are compatible
        patterns = list(groups.keys())
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Two patterns are compatible if their combined pattern has even occurrences
                combined = list(patterns[i]) + list(patterns[j])
                if all(combined.count(x) % 2 == 0 for x in set(combined)):
                    sum1 = sum(nums[k] for k in groups[patterns[i]])
                    sum2 = sum(nums[k] for k in groups[patterns[j]])
                    max_sum = max(max_sum, sum1 + sum2)
                    
        return max_sum