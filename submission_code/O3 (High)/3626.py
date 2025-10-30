class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        """
        Finds the smallest integer ≥ n whose digits' product is divisible by t.
        Constraints are tiny (t ≤ 10), so a straightforward linear search is
        more than fast enough.
        """
        
        # Helper: product of decimal digits of x
        def digit_product(x: int) -> int:
            product = 1
            for ch in str(x):
                d = int(ch)
                if d == 0:           # once a 0 is met, product is 0
                    return 0
                product *= d
            return product
        
        # If t is 1, every number works.
        if t == 1:
            return n
        
        current = n
        while True:
            if digit_product(current) % t == 0:
                return current
            current += 1