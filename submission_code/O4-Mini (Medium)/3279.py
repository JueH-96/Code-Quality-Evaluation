class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Count how many odd and even numbers are in [1, n] and [1, m]
        n_odd = (n + 1) // 2
        n_even = n // 2
        m_odd = (m + 1) // 2
        m_even = m // 2
        
        # Alice wins exactly when x + y is odd,
        # i.e., one of x,y is odd and the other is even.
        return n_odd * m_even + n_even * m_odd