class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if not nums:
            return 0
        
        # Initialize based on the first element's parity
        if nums[0] % 2 == 0:
            even_max = nums[0]
            odd_max = float('-inf')
        else:
            odd_max = nums[0]
            even_max = float('-inf')
        
        for num in nums[1:]:
            if num % 2 == 0:
                # Current number is even
                new_even = max(even_max + num, odd_max + num - x)
                even_max = max(even_max, new_even)
            else:
                # Current number is odd
                new_odd = max(odd_max + num, even_max + num - x)
                odd_max = max(odd_max, new_odd)
        
        return max(even_max, odd_max)