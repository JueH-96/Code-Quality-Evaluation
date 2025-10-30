class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        max_even = 0
        max_odd = 0
        
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                max_even = max(max_even, max_odd + 1)
            else:
                odd_count += 1
                max_odd = max(max_odd, max_even + 1)
        
        longest_alternating = max(max_even, max_odd)
        longest_same_parity = max(even_count, odd_count)
        
        return max(longest_alternating, longest_same_parity)