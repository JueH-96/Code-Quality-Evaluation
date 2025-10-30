class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        even_length = 0  # max length ending with even
        odd_length = 0   # max length ending with odd
        
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                new_even = max(even_length, odd_length + 1)
                even_length = new_even
            else:
                odd_count += 1
                new_odd = max(odd_length, even_length + 1)
                odd_length = new_odd
        
        max_single = max(even_count, odd_count)
        max_alt = max(even_length, odd_length)
        return max(max_single, max_alt)