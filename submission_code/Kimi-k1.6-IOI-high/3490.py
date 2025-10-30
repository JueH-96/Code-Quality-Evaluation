class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        max_even = 0
        max_odd = 0
        
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                new_even = max(max_even, max_odd + 1)
                new_odd = max_odd
                max_even, max_odd = new_even, new_odd
            else:
                odd_count += 1
                new_odd = max(max_odd, max_even + 1)
                new_even = max_even
                max_even, max_odd = new_even, new_odd
        
        candidate1 = max(even_count, odd_count)
        candidate2 = max(max_even, max_odd)
        
        return max(candidate1, candidate2)