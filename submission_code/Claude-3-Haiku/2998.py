class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            if num < 10:
                count += 1
            else:
                num_str = str(num)
                if len(num_str) % 2 == 0:
                    mid = len(num_str) // 2
                    if sum(map(int, num_str[:mid])) == sum(map(int, num_str[mid:])):
                        count += 1
        return count