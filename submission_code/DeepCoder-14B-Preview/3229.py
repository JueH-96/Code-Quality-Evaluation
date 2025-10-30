from bisect import bisect_right

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Generate all possible palindromic numbers up to 1e9
        palindromes = set()
        for length in range(1, 10):
            if length == 1:
                for num in range(1, 10):
                    palindromes.add(num)
                continue
            if length % 2 == 0:
                half = length // 2
                start = 10 ** (half - 1)
                end = (10 ** half) - 1
                for num in range(start, end + 1):
                    s = str(num)
                    palindrome = s + s[::-1]
                    y = int(palindrome)
                    palindromes.add(y)
            else:
                half = (length - 1) // 2
                start = 10 ** (half - 1) if half > 0 else 0
                end = (10 ** half) - 1 if half > 0 else 9
                for num in range(start, end + 1):
                    s = str(num)
                    for middle in '0123456789':
                        palindrome_str = s + middle + s[::-1]
                        y = int(palindrome_str)
                        palindromes.add(y)
        
        # Convert the set to a sorted list for processing
        palindromes = sorted(palindromes)
        if not palindromes:
            return 0  # This case should not occur as we have at least 1-digit palindromes
        
        # Sort the array and compute the prefix sum
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums_sorted[i]
        
        min_cost = float('inf')
        for y in palindromes:
            # Find the rightmost index where elements are <= y
            idx = bisect_right(nums_sorted, y)
            sum_left = prefix[idx]
            sum_right = prefix[n] - sum_left
            # Calculate the cost
            cost = (y * idx - sum_left) + (sum_right - y * (n - idx))
            if cost < min_cost:
                min_cost = cost
        
        return min_cost