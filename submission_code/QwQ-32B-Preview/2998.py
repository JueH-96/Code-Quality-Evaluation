class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            length = len(s)
            if length % 2 != 0:
                continue  # Skip numbers with odd number of digits
            n = length // 2
            sum_first = sum(int(s[i]) for i in range(n))
            sum_second = sum(int(s[i]) for i in range(n, 2 * n))
            if sum_first == sum_second:
                count += 1
        return count