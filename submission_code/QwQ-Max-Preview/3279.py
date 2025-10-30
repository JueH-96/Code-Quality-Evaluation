class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count_even_x = n // 2
        count_odd_x = (n + 1) // 2
        count_odd_y = (m + 1) // 2
        count_even_y = m // 2
        return count_even_x * count_odd_y + count_odd_x * count_even_y