class Solution:
    def _calculate_digit_product(self, num: int) -> int:
        # This helper function calculates the product of the digits of `num`.
        # It is called with `num >= 1` because `current_num` (which is `num` here)
        # starts at `n` (where `n >= 1`) and only increases.

        # Initialize product to 1, as we will be multiplying digits.
        product = 1
        
        # If num is 0, its digit product could be considered 0.
        # However, in this problem's context, num will always be positive.
        # For single-digit numbers (e.g., 7), the loop runs once, and product becomes the digit itself.
        # For numbers like 10, 20, the product becomes 0 due to the '0' digit check.

        # We can modify the local variable `num` directly.
        while num > 0:
            digit = num % 10
            
            if digit == 0:
                # If any digit is 0, the entire product of digits is 0.
                # This is an optimization and correctly handles cases like '10', '20', '100'.
                return 0
            
            product *= digit
            num //= 10  # Integer division to "remove" the last processed digit.
            
        return product

    def smallestNumber(self, n: int, t: int) -> int:
        current_num = n
        
        # We search for the smallest number >= n that satisfies the condition.
        # The loop is guaranteed to terminate because:
        # 1. Constraints: n >= 1, t >= 1.
        # 2. Eventually, `current_num` will reach a number containing the digit '0'
        #    (e.g., 10, 20, ..., 100, 110, etc.).
        # 3. The product of digits for such a number is 0.
        # 4. Since 0 is divisible by any `t` (as t >= 1, so 0 % t == 0),
        #    this number will satisfy the condition if no smaller number already did.
        # Given n <= 100, this termination happens relatively quickly.
        
        while True:
            prod = self._calculate_digit_product(current_num)
            
            # Check if the calculated product of digits is divisible by t.
            # Note: `prod` can be 0 (e.g., for current_num = 10).
            # The expression `0 % t == 0` is true for any integer t >= 1.
            if prod % t == 0:
                return current_num
            
            current_num += 1