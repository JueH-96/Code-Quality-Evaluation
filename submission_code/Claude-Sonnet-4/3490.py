class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Pattern 1: All even numbers
        even_count = sum(1 for x in nums if x % 2 == 0)
        
        # Pattern 2: All odd numbers  
        odd_count = sum(1 for x in nums if x % 2 == 1)
        
        # Pattern 3: Even, Odd, Even, Odd, ...
        pattern3_count = 0
        expecting_even = True
        for num in nums:
            if expecting_even and num % 2 == 0:
                pattern3_count += 1
                expecting_even = False
            elif not expecting_even and num % 2 == 1:
                pattern3_count += 1
                expecting_even = True
        
        # Pattern 4: Odd, Even, Odd, Even, ...
        pattern4_count = 0
        expecting_odd = True
        for num in nums:
            if expecting_odd and num % 2 == 1:
                pattern4_count += 1
                expecting_odd = False
            elif not expecting_odd and num % 2 == 0:
                pattern4_count += 1
                expecting_odd = True
        
        return max(even_count, odd_count, pattern3_count, pattern4_count)