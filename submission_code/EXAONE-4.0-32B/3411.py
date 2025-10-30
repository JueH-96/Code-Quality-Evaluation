import sys
from functools import lru_cache

sys.setrecursionlimit(10000)

class Solution:
    def findProductsOfElements(self, queries):
        answers = []
        for query in queries:
            l, r, mod_val = query
            k = 0
            temp = mod_val
            while temp % 2 == 0:
                k += 1
                temp //= 2
            m_val = temp
            
            S = self.exponent_sum(l, r)
            
            if mod_val == 1:
                answers.append(0)
            elif k == 0:
                answers.append(pow(2, S, mod_val))
            else:
                if S < k:
                    ans = (1 << S) % mod_val
                    answers.append(ans)
                else:
                    if m_val == 1:
                        answers.append(0)
                    else:
                        part_m = pow(2, S, m_val)
                        inv2k = pow(2, -k, m_val)
                        t = (part_m * inv2k) % m_val
                        ans = t * (1 << k)
                        answers.append(ans)
        return answers

    def exponent_sum(self, l, r):
        if l > r:
            return 0
        i1 = self.find_block_index(l)
        i2 = self.find_block_index(r)
        total_exp = 0
        if i1 == i2:
            bits = self.get_bits(i1)
            start_idx = l - self.G(i1-1)
            end_idx = r - self.G(i1-1)
            total_exp = sum(bits[start_idx:end_idx+1])
        else:
            bits1 = self.get_bits(i1)
            start1 = l - self.G(i1-1)
            total_exp += sum(bits1[start1:])
            if i1+1 <= i2-1:
                total_exp += self.H(i2-1) - self.H(i1)
            bits2 = self.get_bits(i2)
            end2 = r - self.G(i2-1)
            total_exp += sum(bits2[:end2+1])
        return total_exp

    def find_block_index(self, x):
        if x == 0:
            return 1
        low, high = 1, 10**18
        while low <= high:
            mid = (low + high) // 2
            g_low = self.G(mid-1)
            g_high = self.G(mid)
            if g_low <= x < g_high:
                return mid
            elif x < g_low:
                high = mid - 1
            else:
                low = mid + 1
        return low

    @lru_cache(maxsize=None)
    def G(self, n):
        if n == 0:
            return 0
        total = 0
        j = 0
        while (1 << j) <= n:
            block_size = 1 << (j+1)
            full_blocks = (n+1) // block_size
            remainder = (n+1) % block_size
            count_j = full_blocks * (1 << j)
            if remainder > (1 << j):
                count_j += remainder - (1 << j)
            total += count_j
            j += 1
        return total

    @lru_cache(maxsize=None)
    def H(self, n):
        if n == 0:
            return 0
        total = 0
        j = 0
        while (1 << j) <= n:
            block_size = 1 << (j+1)
            full_blocks = (n+1) // block_size
            remainder = (n+1) % block_size
            count_j = full_blocks * (1 << j)
            if remainder > (1 << j):
                count_j += remainder - (1 << j)
            total += j * count_j
            j += 1
        return total

    def get_bits(self, i):
        bits = []
        pos = 0
        n = i
        while n:
            if n & 1:
                bits.append(pos)
            n //= 2
            pos += 1
        return bits