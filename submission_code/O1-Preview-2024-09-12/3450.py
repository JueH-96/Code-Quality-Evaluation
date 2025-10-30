class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        T = 2 * (n - 1)
        t_mod_T = k % T
        if t_mod_T <= n - 1:
            return t_mod_T
        else:
            return 2 * (n - 1) - t_mod_T