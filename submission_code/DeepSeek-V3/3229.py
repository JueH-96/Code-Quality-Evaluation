class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Generate all palindromic numbers less than 10^9
        palindromes = set()
        
        # Generate single-digit palindromes
        for i in range(1, 10):
            palindromes.add(i)
        
        # Generate two-digit palindromes
        for i in range(1, 10):
            palindromes.add(i * 10 + i)
        
        # Generate three-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                palindromes.add(i * 100 + j * 10 + i)
        
        # Generate four-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                palindromes.add(i * 1000 + j * 100 + j * 10 + i)
        
        # Generate five-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                for k in range(0, 10):
                    palindromes.add(i * 10000 + j * 1000 + k * 100 + j * 10 + i)
        
        # Generate six-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                for k in range(0, 10):
                    palindromes.add(i * 100000 + j * 10000 + k * 1000 + k * 100 + j * 10 + i)
        
        # Generate seven-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                for k in range(0, 10):
                    for l in range(0, 10):
                        palindromes.add(i * 1000000 + j * 100000 + k * 10000 + l * 1000 + k * 100 + j * 10 + i)
        
        # Generate eight-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                for k in range(0, 10):
                    for l in range(0, 10):
                        palindromes.add(i * 10000000 + j * 1000000 + k * 100000 + l * 10000 + l * 1000 + k * 100 + j * 10 + i)
        
        # Generate nine-digit palindromes
        for i in range(1, 10):
            for j in range(0, 10):
                for k in range(0, 10):
                    for l in range(0, 10):
                        for m in range(0, 10):
                            palindromes.add(i * 100000000 + j * 10000000 + k * 1000000 + l * 100000 + m * 10000 + l * 1000 + k * 100 + j * 10 + i)
        
        # Convert the set to a sorted list
        palindromes = sorted(palindromes)
        
        # Find the median of nums to minimize the cost
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        median = nums_sorted[n // 2]
        
        # Find the closest palindrome to the median
        min_cost = float('inf')
        for p in palindromes:
            cost = 0
            for num in nums:
                cost += abs(num - p)
            if cost < min_cost:
                min_cost = cost
            if min_cost == 0:
                break
        
        return min_cost