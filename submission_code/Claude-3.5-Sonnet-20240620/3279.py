class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0
        for x in range(1, n + 1):
            odd_y = (m + 1) // 2 if x % 2 == 0 else m // 2
            count += odd_y
        return count