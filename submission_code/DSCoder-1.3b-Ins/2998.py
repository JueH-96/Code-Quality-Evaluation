class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for i in range(low, high+1):
            str_i = str(i)
            if str_i[0] != str_i[-1]:
                continue
            if len(str_i) % 2 == 0:
                continue
            count += 1
        return count