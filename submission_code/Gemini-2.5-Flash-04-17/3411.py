from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:

        def countSetBitsUpToN(N: int, j: int) -> int:
            """Counts numbers in [0, N] with j-th bit set."""
            if N < 0: return 0
            N_plus_1 = N + 1
            power_of_2_j = 1 << j
            power_of_2_j_plus_1 = 1 << (j + 1)
            num_full_blocks = N_plus_1 // power_of_2_j_plus_1
            count = num_full_blocks * power_of_2_j
            remainder = N_plus_1 % power_of_2_j_plus_1
            count += max(0, remainder - power_of_2_j)
            return count

        def S(m: int) -> int:
            """Calculates sum(popcount(i) for i=1 to m)."""
            if m <= 0: return 0
            total_popcount_sum = 0
            # Iterate through bit positions. Max relevant bit position for m up to 10^14 is ~46.
            # Iterate up to 60 to be safe (covers m up to 2^61-1).
            for j in range(61):
                total_popcount_sum += countSetBitsUpToN(m, j)
            return total_popcount_sum

        def SumExponentUpToM(M: int) -> int:
            """Calculates sum of j for each set bit j in numbers from 1 to M."""
            if M <= 0: return 0
            total_exp_sum = 0
            # Iterate through bit positions. Max relevant j for M up to 10^14 is ~46.
            # Iterate up to 60 to be safe.
            for j in range(61):
                total_exp_sum += j * countSetBitsUpToN(M, j)
            return total_exp_sum

        def SumExp(N: int) -> int:
            """Calculates sum(get_exponent(k) for k=0 to N)."""
            if N < 0: return 0

            # Find m_N such that S(m_N - 1) <= N < S(m_N).
            # Binary search for smallest m such that S(m) > N.
            # The range for m is approximately [1, 10^14]. A safe upper bound is 10^14 + 100.
            low_m, high_m = 1, 10**14 + 100
            m_N = high_m # Initialize m_N

            # Find smallest m_N such that S(m_N) > N using binary search
            # Standard binary search to find the first element > target
            low_bs, high_bs = 1, 10**14 + 100
            while low_bs <= high_bs:
                mid_m = low_bs + (high_bs - low_bs) // 2
                current_S = S(mid_m)

                if current_S > N:
                    m_N = mid_m
                    high_bs = mid_m - 1
                else: # S(mid_m) <= N
                    low_bs = mid_m + 1

            # Now m_N is the smallest m such that S(m) > N.
            # Index N falls within powerful_array(m_N).
            # S(m_N - 1) <= N < S(m_N).
            # The sum of exponents up to index S(m_N - 1) - 1 is SumExponentUpToM(m_N - 1).
            sum_before_m_N = SumExponentUpToM(m_N - 1)

            # Index N is the (p_N + 1)-th element (0-indexed p_N) in powerful_array(m_N).
            # The elements in powerful_array(m_N) are powers of 2^j for set bits j in m_N, sorted by j.
            p_N = N - S(m_N - 1)

            # Sum of exponents for the first p_N + 1 elements in powerful_array(m_N)
            sum_in_m_N = 0
            set_bit_count = 0 # 0-indexed count of set bits encountered so far
            # Iterate through bit positions j of m_N from 0 upwards
            # We need the sum of j for the first p_N + 1 set bits (at relative indices 0 to p_N).
            # m_N can be up to ~10^14, requires up to ~47 bits. Max j needed is ~46.
            # Iterate up to bit 60 to be safe.
            for j in range(61):
                if (m_N >> j) & 1: # If j-th bit is set
                    # This j is the exponent for the element at relative index `set_bit_count`.
                    if set_bit_count <= p_N: # We need to include this exponent in the sum
                        sum_in_m_N += j

                    set_bit_count += 1
                    # If we have just processed the element at relative index p_N, we can stop.
                    # The next element would be at relative index p_N + 1, which is not included.
                    if set_bit_count > p_N:
                         break

            return sum_before_m_N + sum_in_m_N

        def count_trailing_zeros(n: int) -> int:
            """Counts the number of trailing zero bits in n."""
            if n == 0: return 0 # Constraint mod_i >= 1
            count = 0
            # n >= 1 based on constraints
            while (n & 1) == 0:
                 count += 1
                 n >>= 1
            return count

        def phi(n: int) -> int:
            """Calculates Euler's totient function phi(n)."""
            if n == 0: return 0
            result = n
            p = 2
            while p * p <= n:
                if n % p == 0:
                    while n % p == 0:
                        n //= p
                    result -= result // p
                p += 1
            if n > 1:
                result -= result // n
            return result

        ans = []
        for from_i, to_i, mod_i in queries:
            m = mod_i
            v = count_trailing_zeros(m)
            m_prime = m >> v # m // (1 << v)

            # Calculate the total exponent E = sum(get_exponent(k) for k=from_i to to_i)
            # SumExp(N) calculates sum for k=0 to N.
            # Sum for k=from_i to to_i is SumExp(to_i) - SumExp(from_i - 1)
            total_exponent = SumExp(to_i) - SumExp(from_i - 1)

            result = 0
            if total_exponent < v:
                # If E < v, 2^E % m = 2^E.
                # Since E < v, 2^E < 2^v. Also m = 2^v * m', so 2^v <= m (unless m=0).
                # Thus 2^E < m. So 2^E % m = 2^E.
                # pow(2, total_exponent, m) correctly calculates 2^total_exponent % m.
                 result = pow(2, total_exponent, m)
            else:
                # If E >= v, we need 2^E % m = (2^(E-v) * 2^v) % (m' * 2^v)
                # This is equivalent to (2^(E-v) % m') * 2^v.
                # Calculate Y = 2^(E-v) % m'.
                E_minus_v = total_exponent - v

                if m_prime == 1: # m is a power of 2, m = 2^v
                    # If E >= v, 2^E is divisible by 2^v, so 2^E % 2^v = 0.
                    result = 0
                else: # m_prime > 1, gcd(2, m') = 1
                    # Calculate Y = pow(2, E_minus_v, m_prime).
                    # Python's pow handles large exponents correctly using modular exponentiation property:
                    # a^b % n = a^(b % phi(n)) % n for b >= phi(n) and gcd(a, n) = 1.
                    Y = pow(2, E_minus_v, m_prime)
                    # The result is Y * 2^v.
                    # Note that Y < m_prime. Y * 2^v < m_prime * 2^v = m.
                    # So (Y * (1 << v)) % m is just Y * (1 << v).
                    result = Y * (1 << v)

            ans.append(result)

        return ans