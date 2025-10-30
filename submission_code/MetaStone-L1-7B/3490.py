class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        a = max(even_count, odd_count)
        
        even_start = 0
        odd_start = 0
        for num in nums:
            if num % 2 == 0:
                even_start = max(even_start, odd_start + 1)
            else:
                odd_start = max(odd_start, even_start + 1)
        b = max(even_start, odd_start)
        
        return max(a, b)