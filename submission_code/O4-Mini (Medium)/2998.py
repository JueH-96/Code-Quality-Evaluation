class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            n = len(s)
            # Only even-digit numbers can be symmetric
            if n % 2 != 0:
                continue
            half = n // 2
            first_sum = sum(int(ch) for ch in s[:half])
            second_sum = sum(int(ch) for ch in s[half:])
            if first_sum == second_sum:
                count += 1
        return count