from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits_upto(n):
            total = 0
            b = 0
            while (1 << b) <= n:
                x = 1 << b
                next_x = x << 1
                count = (n + 1) // next_x * x
                rem = (n + 1) % next_x
                if rem > x:
                    count += rem - x
                total += count
                b += 1
            return total

        def sum_exponents_upto(n):
            total = 0
            b = 0
            while (1 << b) <= n:
                x = 1 << b
                next_x = x << 1
                count = (n + 1) // next_x * x
                rem = (n + 1) % next_x
                if rem > x:
                    count += rem - x
                total += count * b
                b += 1
            return total

        def get_bits(n):
            bits = []
            b = 0
            while (1 << b) <= n:
                if n & (1 << b):
                    bits.append(b)
                b += 1
            return bits

        def find_N(pos):
            low = 1
            high = pos
            res = pos
            while low <= high:
                mid = (low + high) // 2
                s = count_set_bits_upto(mid)
                if s < pos:
                    low = mid + 1
                else:
                    high = mid - 1
                    res = mid
            return res

        def get_exponent_sum(K):
            if K < 0:
                return 0
            pos = K + 1
            N = find_N(pos)
            s_prev = count_set_bits_upto(N - 1)
            offset = (pos - s_prev - 1)
            bits = get_bits(N)
            partial_sum = sum(bits[:offset + 1])
            total = sum_exponents_upto(N - 1) + partial_sum
            return total

        answer = []
        for q in queries:
            L, R, mod = q
            if L > R:
                answer.append(0)
                continue
            sum_LR = get_exponent_sum(R) - get_exponent_sum(L - 1)
            res = pow(2, sum_LR, mod)
            answer.append(res)
        return answer