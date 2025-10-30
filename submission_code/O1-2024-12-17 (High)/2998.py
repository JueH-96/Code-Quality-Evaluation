class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            # Only consider numbers with an even number of digits
            if len(s) % 2 == 0:
                half = len(s) // 2
                # Compare the sum of the first half of digits to the sum of the second half
                if sum(int(d) for d in s[:half]) == sum(int(d) for d in s[half:]):
                    count += 1
        return count