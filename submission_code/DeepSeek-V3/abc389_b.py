class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        count = 0
        for num in range(l, r + 1):
            digits = [int(d) for d in str(num)]
            sum_digits = sum(digits)
            if sum_digits == 0:
                continue  # impossible for positive numbers, but handle to avoid division by zero
            product = 1
            for d in digits:
                product *= d
            if product % sum_digits == 0:
                count += 1
        return count