class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        a = 1
        b = 0
        withset = cost = 0
        bits = 0

        def get(x, i):
            return x & (1 << i) != 0

        while a > 0 and cost <= k:
            n = min(((k + cost - 1) // cost) - b, a)
            if n <= 0:
                a <<= 1
                e = 1 << (x * bits)
                b = (b - e) * 2 + e
                bits += 1
                if bits >= 60:
                    bits += x // 60
                continue
            m = (a - 1) & ~((a - n) - 1)
            add = 0
            while m > 0:
                i = 0
                while get(m, i):
                    add += a ^ (1 << i)
                    i += 1
                m //= 2
            cost += add
            withset |= n
            a = a - n
            last = 0
            a <<= 1
            e = 1 << (x * bits)
            b = (b - e) * 2 + e
            if last:
                add = 0
                m = b
                while m > 0:
                    i = 0
                    while get(m, i):
                        add += last ^ (1 << i)
                        i += 1
                    m //= 2
                cost += add
            last = a ^ n
            bits += 1
            if bits == 60:
                bits += x // 60
        return withset