class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        def digitDifference(num: int) -> int:
            numStr = str(num)
            length = len(numStr)
            digitSum = [0] * 10
            for digit in numStr:
                digitSum[int(digit)] += 1
            return sum(i * (length - i) * 10**(length - 1) for i in range(1, 10)) + sum(digitSum[i] * (length - digitSum[i]) for i in range(10))

        return sum(digitDifference(num) for num in nums)