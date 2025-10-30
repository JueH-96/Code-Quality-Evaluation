class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        n = int(s, 2)
        mod = 10 ** 9 + 7
        
        def count_set_bits(x):
            count = 0
            while x:
                count += x & 1
                x >>= 1
            return count
        
        def is_k_reducible(x):
            for _ in range(k):
                x = count_set_bits(x)
                if x == 1:
                    return True
            return False
        
        count = 0
        for i in range(1, n):
            if is_k_reducible(i):
                count += 1
        
        return count % mod