class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def get_square_free(x):
            if x == 1:
                return 1
            square_free = 1
            # Handle factor 2
            count = 0
            while x % 2 == 0:
                count += 1
                x = x // 2
            if count % 2 == 1:
                square_free *= 2
            # Handle odd factors
            i = 3
            while i * i <= x:
                count = 0
                while x % i == 0:
                    count += 1
                    x = x // i
                if count % 2 == 1:
                    square_free *= i
                i += 2
            # Handle remaining prime factor greater than sqrt(x)
            if x > 1:
                square_free *= x
            return square_free
        
        groups = defaultdict(int)
        max_sum = 0
        for num in nums:
            square_free = get_square_free(num)
            groups[square_free] += num
            if groups[square_free] > max_sum:
                max_sum = groups[square_free]
        return max_sum