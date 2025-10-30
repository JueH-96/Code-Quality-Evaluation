class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(max(11, low), high + 1):
            digits = [int(d) for d in str(num)]
            if len(digits) % 2 == 0:
                mid = len(digits) // 2
                if sum(digits[:mid]) == sum(digits[mid:]):
                    count += 1
        return count