class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Helper functions to compute the sum of odd and even row indices up to h
        def sum_of_odd_indices(h):
            # Number of odd rows is (h+1)//2
            # The sum of the first k odd numbers is k^2
            k = (h + 1) // 2
            return k * k
        
        def sum_of_even_indices(h):
            # Number of even rows is h//2
            # The sum of the first k even numbers is k * (k + 1)
            k = h // 2
            return k * (k + 1)
        
        max_height = 0
        
        # We'll check all heights from 1 up to 20 (enough to cover sum up to 200)
        for h in range(1, 21):
            odd_sum = sum_of_odd_indices(h)
            even_sum = sum_of_even_indices(h)
            
            # Pattern 1: odd rows use red, even rows use blue
            if odd_sum <= red and even_sum <= blue:
                max_height = max(max_height, h)
            
            # Pattern 2: odd rows use blue, even rows use red
            if odd_sum <= blue and even_sum <= red:
                max_height = max(max_height, h)
        
        return max_height