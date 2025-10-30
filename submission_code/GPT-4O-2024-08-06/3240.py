class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price(num, x):
            # Calculate the price of a number based on the given x
            s = bin(num)[2:]  # Get binary representation without '0b' prefix
            s = s[::-1]  # Reverse to make it 1-indexed from right to left
            return sum(1 for i in range(len(s)) if (i + 1) % x == 0 and s[i] == '1')

        total_price = 0
        num = 0

        # Iterate until the total price exceeds k
        while total_price <= k:
            num += 1
            total_price += price(num, x)

        # Since the loop exits when total_price exceeds k, we return num - 1
        return num - 1