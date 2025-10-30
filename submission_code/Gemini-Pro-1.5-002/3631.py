class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        
        @lru_cache(None)
        def is_k_reducible(num, k):
            if num == 1:
                return True
            if k == 0:
                return False
            
            set_bits = bin(num).count('1')
            return is_k_reducible(set_bits, k - 1)

        count = 0
        for i in range(1, n):
            if is_k_reducible(i, k):
                count += 1
        
        return count