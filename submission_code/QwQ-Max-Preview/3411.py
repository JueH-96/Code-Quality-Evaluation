class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def sum_bits(n):
            if n == 0:
                return 0
            res = 0
            k = 0
            while (1 << k) <= n:
                m = 1 << k
                two_m = m << 1
                full_cycles = (n + 1) // two_m
                count = full_cycles * m
                remainder = (n + 1) % two_m
                count += max(0, remainder - m)
                res += count
                k += 1
            return res

        def count_set_bits(k, x):
            if x < 0:
                return 0
            cycle = 1 << (k + 1)
            count = (x + 1) // cycle * (1 << k)
            remainder = (x + 1) % cycle
            count += max(0, remainder - (1 << k))
            return count

        def find_i(pos):
            if pos < 0:
                return -1
            low = 1
            high = pos + 1
            best = -1
            while low <= high:
                mid = (low + high) // 2
                s_mid = sum_bits(mid - 1)
                if s_mid <= pos:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        def get_exponents(i):
            exponents = []
            pos = 0
            while i > 0:
                if i & 1:
                    exponents.append(pos)
                i >>= 1
                pos += 1
            return exponents

        answer = []
        for query in queries:
            L, R, mod = query
            if L > R:
                answer.append(0 % mod)
                continue
            i_start = find_i(L)
            i_end = find_i(R)
            if i_start is None or i_end is None:
                answer.append(0 % mod)
                continue
            sum_t = 0
            # Process i_start
            S_i_start = sum_bits(i_start - 1)
            E_i_start = sum_bits(i_start) - 1
            a = max(S_i_start, L)
            b = min(E_i_start, R)
            if a <= b:
                s = a - S_i_start
                e = b - S_i_start
                exponents = get_exponents(i_start)
                if exponents:
                    s = max(s, 0)
                    e = min(e, len(exponents) - 1)
                    if s <= e:
                        sum_t += sum(exponents[s:e + 1])
            # Process i_end if different from i_start
            if i_start != i_end:
                S_i_end = sum_bits(i_end - 1)
                E_i_end = sum_bits(i_end) - 1
                a_end = max(S_i_end, L)
                b_end = min(E_i_end, R)
                if a_end <= b_end:
                    s_end = a_end - S_i_end
                    e_end = b_end - S_i_end
                    exponents_end = get_exponents(i_end)
                    if exponents_end:
                        s_end = max(s_end, 0)
                        e_end = min(e_end, len(exponents_end) - 1)
                        if s_end <= e_end:
                            sum_t += sum(exponents_end[s_end:e_end + 1])
                # Process bulk between i_start +1 and i_end -1
                a_bulk = i_start + 1
                b_bulk = i_end - 1
                if a_bulk <= b_bulk:
                    sum_bulk = 0
                    max_k = 60  # Sufficiently large to cover exponents up to 2^60
                    for k in range(max_k + 1):
                        cnt = count_set_bits(k, b_bulk) - count_set_bits(k, a_bulk - 1)
                        sum_bulk += k * cnt
                    sum_t += sum_bulk
            # Compute answer
            if mod == 1:
                answer.append(0)
            else:
                answer.append(pow(2, sum_t, mod))
        return answer