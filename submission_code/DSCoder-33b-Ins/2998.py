class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for i in range(low, high+1):
            if len(str(i)) % 2 == 0:
                mid = len(str(i)) // 2
                first_half = str(i)[:mid]
                second_half = str(i)[mid:]
                if sum(map(int, first_half)) == sum(map(int, second_half)):
                    count += 1
        return count