from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        self._memo_csb = {0: 0}
        self._memo_tes = {0: 0}
        
        ans = []
        for fro, to, mod in queries:
            if mod == 1:
                ans.append(0)
                continue

            total_s = self._calculate_total_exponent(to) - self._calculate_total_exponent(fro - 1)

            phi_mod = self._phi(mod)
            exp = total_s
            # For b >= phi(m), a^b === a^(phi(m) + b % phi(m)) (mod m)
            # This handles cases where base and modulus are not coprime.
            if exp >= phi_mod:
                exp = phi_mod + (total_s % phi_mod)
            
            result = pow(2, exp, mod)
            ans.append(result)
        return ans

    def _count_set_bits_up_to(self, n: int) -> int:
        """Calculates sum of popcounts for numbers from 1 to n."""
        if n in self._memo_csb:
            return self._memo_csb[n]
        
        k = n.bit_length() - 1
        p = 1 << k
        
        res = k * (p >> 1) + (n - p + 1) + self._count_set_bits_up_to(n - p)
        self._memo_csb[n] = res
        return res

    def _count_set_bits_at_pos(self, n: int, j: int) -> int:
        """Counts how many numbers from 1 to n have the j-th bit set."""
        if n < 0: return 0
        p = 1 << j
        period = p << 1
        
        num_periods = (n + 1) // period
        count = num_periods * p
        
        rem = (n + 1) % period
        count += max(0, rem - p)
        return count

    def _total_exponent_sum_up_to(self, n: int) -> int:
        """Calculates sum of exponents for all powerful arrays from 1 to n."""
        if n in self._memo_tes:
            return self._memo_tes[n]
        
        total = 0
        # n is up to ~2*10^15, bit_length ~ 51. 60 is a safe upper bound.
        for j in range(60):
            total += j * self._count_set_bits_at_pos(n, j)
        
        self._memo_tes[n] = total
        return total

    def _find_x(self, k: int) -> int:
        """Finds x such that big_nums[k] is in the powerful array of x."""
        target_len = k + 1
        
        low = 1
        high = 2 * 10**15 
        ans_x = high
        
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                low = 1
                continue
            if self._count_set_bits_up_to(mid) >= target_len:
                ans_x = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans_x

    def _calculate_total_exponent(self, k: int) -> int:
        """Calculates sum of exponents of big_nums from index 0 to k."""
        if k < 0:
            return 0
        
        x = self._find_x(k)
        
        len_before = self._count_set_bits_up_to(x - 1)
        sum_full = self._total_exponent_sum_up_to(x - 1)
        
        rem_elems = k - len_before + 1
        
        sum_partial = 0
        elems_found = 0
        for j in range(60):
            if (x >> j) & 1:
                if elems_found < rem_elems:
                    sum_partial += j
                    elems_found += 1
                else:
                    break
        return sum_full + sum_partial

    def _phi(self, n: int) -> int:
        """Calculates Euler's totient function."""
        res = n
        p = 2
        temp_n = n
        while p * p <= temp_n:
            if temp_n % p == 0:
                res = res // p * (p - 1)
                while temp_n % p == 0:
                    temp_n //= p
            p += 1
        if temp_n > 1:
            res = res // temp_n * (temp_n - 1)
        return res