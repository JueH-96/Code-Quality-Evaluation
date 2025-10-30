class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        """
        Return the smallest integer >= n such that the product of its digits is divisible by t.
        """
        def digit_product(x: int) -> int:
            prod = 1
            for ch in str(x):
                prod *= int(ch)
            return prod
        
        # If t == 1, every product is divisible by 1
        if t == 1:
            return n
        
        x = n
        while True:
            if digit_product(x) % t == 0:
                return x
            x += 1