class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def cnt_sum(x):
            if x == 0:
                return 0
            total = 0
            for i in range(60):
                mask = 1 << i
                if mask > x:
                    break
                period = mask << 1
                full_cycles = (x + 1) // period
                cnt = full_cycles * mask
                remainder = (x + 1) % period
                cnt += max(0, remainder - mask)
                total += cnt
            return total

        def exp_sum(x):
            if x == 0:
                return 0
            total = 0
            for i in range(60):
                mask = 1 << i
                if mask > x:
                    break
                period = mask << 1
                full_cycles = (x + 1) // period
                cnt = full_cycles * mask
                remainder = (x + 1) % period
                cnt += max(0, remainder - mask)
                total += cnt * i
            return total

        def get_exponents(x):
            exponents = []
            i = 0
            while x > 0:
                if x & 1:
                    exponents.append(i)
                x >>= 1
                i += 1
            return exponents

        def binary_search_k(k):
            if k < 0:
                return 0
            low = 1
            high = 1 << 60
            x = 0
            while low <= high:
                mid = (low + high) // 2
                s_mid_minus1 = cnt_sum(mid - 1)
                s_mid = cnt_sum(mid)
                if s_mid_minus1 <= k < s_mid:
                    x = mid
                    break
                elif s_mid <= k:
                    low = mid + 1
                else:
                    high = mid - 1
            else:
                x = low
                s_x_minus1 = cnt_sum(x - 1)
                s_x = cnt_sum(x)
                if not (s_x_minus1 <= k < s_x):
                    x = high
            return x

        def sum_exp(k):
            if k < 0:
                return 0
            x = binary_search_k(k)
            s_x_minus1 = cnt_sum(x - 1)
            p = k - s_x_minus1
            exponents = get_exponents(x)
            if p >= len(exponents):
                p = len(exponents) - 1
            sum_p = sum(exponents[:p+1])
            return exp_sum(x - 1) + sum_p

        answer = []
        for a, b, mod in queries:
            total = sum_exp(b) - sum_exp(a - 1)
            result = pow(2, total, mod) if mod != 1 else 0
            answer.append(result)
        return answer