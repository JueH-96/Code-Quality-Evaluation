class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        even_x = n // 2
        odd_x = n - even_x
        
        even_y = m // 2
        odd_y = m - even_y
        
        return even_x * odd_y + odd_x * even_y