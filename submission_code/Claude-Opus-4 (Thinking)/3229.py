class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            s = str(n)
            return s == s[::-1]
        
        def calculate_cost(nums, target):
            return sum(abs(x - target) for x in nums)
        
        # Sort array to find median
        nums.sort()
        n = len(nums)
        
        # Find median
        if n % 2 == 1:
            median = nums[n // 2]
        else:
            median = (nums[n // 2 - 1] + nums[n // 2]) // 2
        
        # Find palindromic candidates around median
        candidates = []
        
        # Check if median itself is palindromic
        if is_palindrome(median):
            candidates.append(median)
        
        # Search for palindromes below median
        left = median - 1
        found_left = 0
        while left >= 1 and found_left < 10:
            if is_palindrome(left):
                candidates.append(left)
                found_left += 1
            left -= 1
        
        # Search for palindromes above median
        right = median + 1
        found_right = 0
        while right < 10**9 and found_right < 10:
            if is_palindrome(right):
                candidates.append(right)
                found_right += 1
            right += 1
        
        # If we haven't found enough candidates, expand search
        if len(candidates) < 5:
            # Try a wider range
            search_range = max(1000, abs(nums[-1] - nums[0]))
            for num in range(max(1, median - search_range), min(10**9, median + search_range + 1)):
                if is_palindrome(num):
                    candidates.append(num)
                if len(candidates) >= 20:
                    break
        
        # Calculate minimum cost among all candidates
        min_cost = float('inf')
        for candidate in candidates:
            cost = calculate_cost(nums, candidate)
            min_cost = min(min_cost, cost)
        
        return min_cost