class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits(x):
            res = 0
            for k in range(64):
                two = 1 << k
                next_two = two << 1
                high = (x + 1) // next_two
                res += high * two
                remainder = (x + 1) % next_two
                res += max(0, remainder - two)
            return res
        
        def sum_positions_upto(x):
            res = 0
            for k in range(64):
                two = 1 << k
                next_two = 1 << (k + 1)
                high = (x + 1) // next_two
                res += high * two * k
                remainder = (x + 1) % next_two
                res += max(0, remainder - two) * k
            return res
        
        def find_i(k):
            if k < 0:
                return 0
            low = 1
            high = 1
            while True:
                current = count_set_bits(high)
                if current > k:
                    break
                high *= 2
            while low < high:
                mid = (low + high) // 2
                mid_count = count_set_bits(mid)
                if mid_count > k:
                    high = mid
                else:
                    low = mid + 1
            return low
        
        def sum_exponents_up_to(k):
            if k < 0:
                return 0
            i = find_i(k)
            sum_pos = sum_positions_upto(i - 1)
            pos = k - count_set_bits(i - 1)
            sum_exp = 0
            count = 0
            for bit in range(64):
                if (i >> bit) & 1:
                    if count <= pos:
                        sum_exp += bit
                        count += 1
                    else:
                        break
            return sum_pos + sum_exp
        
        result = []
        for q in queries:
            L, R, mod = q
            if mod == 1:
                result.append(0)
                continue
            if L > R:
                result.append(1 % mod)
                continue
            sum_total = sum_exponents_up_to(R) - sum_exponents_up_to(L - 1)
            ans = pow(2, sum_total, mod)
            result.append(ans)
        
        return result