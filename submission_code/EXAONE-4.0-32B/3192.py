import random

class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        
        if n == 0:
            return (a * b) % mod
        
        if n <= 25:
            best = 0
            full = (1 << n) - 1
            for x in range(0, full + 1):
                ax = a ^ x
                bx = b ^ x
                product = ax * bx
                if product > best:
                    best = product
            return best % mod
        else:
            full = (1 << n) - 1
            a_low = a & full
            b_low = b & full
            candidates = {
                0,
                full,
                a_low,
                b_low,
                a_low ^ b_low,
                a_low | b_low,
                a_low & b_low,
                a_low ^ full,
                b_low ^ full,
                (a_low | b_low) ^ full,
                full // 2,
                full // 2 + 1
            }
            for _ in range(10000):
                x = random.randint(0, full)
                candidates.add(x)
            
            best = 0
            for x in candidates:
                ax = a ^ x
                bx = b ^ x
                product = ax * bx
                if product > best:
                    best = product
            return best % mod