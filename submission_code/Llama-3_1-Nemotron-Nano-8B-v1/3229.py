from typing import List
import bisect

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        total_sum = prefix[-1]
        
        # Determine medians
        if n % 2 == 1:
            medians = [nums[n // 2]]
        else:
            medians = [nums[(n // 2) - 1], nums[n // 2]]
        
        # Generate candidate palindromes
        candidates = set()
        for m in medians:
            if self.is_palindrome(m):
                candidates.add(m)
            # Next higher palindrome
            np = self.next_palindrome(m)
            candidates.add(np)
            # Previous lower palindrome
            pp = self.previous_palindrome(m)
            if pp is not None:
                candidates.add(pp)
        
        # Calculate minimum cost
        min_cost = float('inf')
        for y in candidates:
            # Find the partition point using binary search
            idx = bisect.bisect_right(nums, y)
            sum_left = y * idx - prefix[idx]
            sum_right = (prefix[n] - prefix[idx]) - y * (n - idx)
            total_cost = sum_left + sum_right
            if total_cost < min_cost:
                min_cost = total_cost
        
        return min_cost
    
    def is_palindrome(self, x: int) -> bool:
        s = str(x)
        return s == s[::-1]
    
    def next_palindrome(self, x: int) -> int:
        x += 1
        while True:
            if self.is_palindrome(x):
                return x
            x += 1
    
    def previous_palindrome(self, x: int) -> int:
        if x < 11:
            return None
        x -= 1
        while x >= 0:
            if self.is_palindrome(x):
                return x
            x -= 1
        return None