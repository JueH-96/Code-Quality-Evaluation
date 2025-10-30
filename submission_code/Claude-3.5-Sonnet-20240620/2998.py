class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(num):
            str_num = str(num)
            length = len(str_num)
            if length % 2 != 0:
                return False
            mid = length // 2
            return sum(int(d) for d in str_num[:mid]) == sum(int(d) for d in str_num[mid:])

        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
        return count