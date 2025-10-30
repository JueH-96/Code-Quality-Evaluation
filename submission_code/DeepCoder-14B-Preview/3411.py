class Solution:
    def count_set_bits(self, n):
        return bin(n).count('1')

    def compute_S(self, i):
        res = 0
        for k in range(61):
            pow2k = 1 << k
            pow2k1 = pow2k << 1
            higher = (i + 1) // pow2k1
            lower = (i + 1) % pow2k1
            if lower > pow2k:
                lower -= pow2k
            else:
                lower = 0
            res += higher * pow2k + lower
        return res

    def findProductsOfElements(self, queries: list) -> list:
        answers = []
        for query in queries:
            l, r, mod = query
            if mod == 1:
                answers.append(0)
                continue

            # Binary search for i_min: minimal i where compute_S(i) >= l + 1
            left, right = 1, 2 * (10**18)
            i_min = None
            while left <= right:
                mid = (left + right) // 2
                s = self.compute_S(mid)
                if s >= l + 1:
                    i_min = mid
                    right = mid - 1
                else:
                    left = mid + 1

            # Binary search for i_max: maximal i where compute_S(i-1) <= r
            left, right = 1, 2 * (10**18)
            i_max = None
            while left <= right:
                mid = (left + right) // 2
                if mid == 0:
                    s = 0
                else:
                    s = self.compute_S(mid - 1)
                if s <= r:
                    i_max = mid
                    left = mid + 1
                else:
                    right = mid - 1

            if i_min is None or i_max is None or i_min > i_max:
                S = 0
            else:
                S = 0
                for i in range(i_min, i_max + 1):
                    start_i = self.compute_S(i - 1)
                    end_i = self.compute_S(i) - 1
                    a = max(l, start_i)
                    b = min(r, end_i)
                    if a > b:
                        continue
                    t = b - a + 1
                    # Generate exp_list
                    exp_list = []
                    for k in range(61):
                        if (i >> k) & 1:
                            exp_list.append(k)
                    exp_list.sort()
                    p = a - start_i
                    if p < 0 or p >= len(exp_list):
                        continue
                    num_elements = min(t, len(exp_list) - p)
                    sum_i = sum(exp_list[p : p + num_elements])
                    S += sum_i

            if S == 0:
                product_mod = 1 % mod
            else:
                product_mod = pow(2, S, mod)
            answers.append(product_mod)
        return answers