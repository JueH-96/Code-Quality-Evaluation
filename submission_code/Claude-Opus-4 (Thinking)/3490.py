class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Pattern 1: All even numbers
        count_all_even = sum(1 for num in nums if num % 2 == 0)
        
        # Pattern 2: All odd numbers
        count_all_odd = sum(1 for num in nums if num % 2 == 1)
        
        # Pattern 3: Alternating starting with even
        count_alt_even = 0
        next_parity = 0  # Looking for even first
        for num in nums:
            if num % 2 == next_parity:
                count_alt_even += 1
                next_parity = 1 - next_parity
        
        # Pattern 4: Alternating starting with odd
        count_alt_odd = 0
        next_parity = 1  # Looking for odd first
        for num in nums:
            if num % 2 == next_parity:
                count_alt_odd += 1
                next_parity = 1 - next_parity
        
        return max(count_all_even, count_all_odd, count_alt_even, count_alt_odd)