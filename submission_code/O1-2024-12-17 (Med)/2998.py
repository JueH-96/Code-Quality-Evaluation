class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            # Check if number of digits is even
            if len(s) % 2 == 0:
                half = len(s) // 2
                # Split into two halves
                left, right = s[:half], s[half:]
                # Sum the digits in both halves
                sum_left = sum(int(d) for d in left)
                sum_right = sum(int(d) for d in right)
                # Check if sums are equal
                if sum_left == sum_right:
                    count += 1
        return count