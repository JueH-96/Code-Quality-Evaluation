class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        cnt = 0
        for x in range(low, high + 1):
            x_str = str(x)
            if len(x_str) % 2 != 0:
                continue
            n = len(x_str) // 2
            first_half = x_str[:n]
            second_half = x_str[n:]
            sum_first = sum(int(d) for d in first_half)
            sum_second = sum(int(d) for d in second_half)
            if sum_first == sum_second:
                cnt += 1
        return cnt