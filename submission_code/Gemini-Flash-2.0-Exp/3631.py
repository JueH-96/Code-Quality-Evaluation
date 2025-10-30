class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7

        def count_set_bits(x):
            count = 0
            while x > 0:
                x &= (x - 1)
                count += 1
            return count

        def is_k_reducible(x, k):
            for _ in range(k):
                x = count_set_bits(x)
                if x == 1:
                    return True
            return x == 1

        num = int(s, 2)
        count = 0
        for i in range(1, num):
            if is_k_reducible(i, k):
                count += 1
        return count % MOD