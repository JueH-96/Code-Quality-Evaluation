class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        x_e = n // 2
        x_o = n - x_e
        y_e = m // 2
        y_o = m - y_e
        return x_e * y_o + x_o * y_e