class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize even_max and odd_max based on the first element
        if nums[0] % 2 == 0:
            even_max = nums[0]
            odd_max = -float('inf')
        else:
            even_max = -float('inf')
            odd_max = nums[0]
        
        for i in range(1, n):
            num = nums[i]
            if num % 2 == 0:
                # Compute new_even
                new_even = even_max
                if even_max != -float('inf'):
                    new_even = max(new_even, even_max + num)
                if odd_max != -float('inf'):
                    new_even = max(new_even, odd_max + num - x)
                # new_odd remains the same
                new_odd = odd_max
            else:
                # Compute new_odd
                new_odd = odd_max
                if even_max != -float('inf'):
                    new_odd = max(new_odd, even_max + num - x)
                if odd_max != -float('inf'):
                    new_odd = max(new_odd, odd_max + num)
                # new_even remains the same
                new_even = even_max
            
            # Update even_max and odd_max
            even_max, odd_max = new_even, new_odd
        
        return max(even_max, odd_max)