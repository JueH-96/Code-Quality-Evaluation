class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        o_x = (n + 1) // 2  # Number of odd x in [1,n]
        e_x = n // 2        # Number of even x in [1,n]
        o_y = (m + 1) // 2  # Number of odd y in [1,m]
        e_y = m // 2        # Number of even y in [1,m]
        return o_x * e_y + e_x * o_y