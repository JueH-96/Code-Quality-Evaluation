class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        total_price = 0
        num = 0
        
        while total_price <= k:
            num += 1
            price = 0
            
            # Calculate the price of the current number
            for i in range(1, num.bit_length() + 1):
                if (num >> (i - 1)) & 1 and i % x == 0:
                    price += 1
            
            total_price += price
        
        return num - 1