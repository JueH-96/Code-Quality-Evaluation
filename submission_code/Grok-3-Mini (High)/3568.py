class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        key_num = 0
        for place in [1000, 100, 10, 1]:
            digit_min = min((num1 // place) % 10, (num2 // place) % 10, (num3 // place) % 10)
            key_num = key_num * 10 + digit_min
        return key_num