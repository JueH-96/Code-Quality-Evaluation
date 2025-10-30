import math
from typing import List

class Solution:
    # Helper to calculate population count (number of set bits)
    # n <= 10^15, so max ~50 bits. This is fast enough.
    def _popcount(self, n: int) -> int:
        count = 0
        while n > 0:
            n &= (n - 1)  # Brian Kernighan's algorithm
            count += 1
        return count

    # Helper to calculate the sum of popcounts for numbers from 0 to N.
    # This is also C(N) in the problem description (sum of lengths of powerful arrays up to N).
    # sum_{i=0 to N} popcount(i)
    # N <= 10^15. Iterates log N times (approx 50).
    def _count_total_elements_up_to_N(self, N: int) -> int:
        if N < 0:
            return 0
        count = 0
        k = 0 # Current bit position
        # Iterate through bit positions from 0 up to max_bit in N (~50 for 10^15)
        # The loop condition (1 << k) <= N + 1 is key for handling ranges correctly.
        while (1 << k) <= N + 1:
            period = 1 << (k + 1) # Cycle length for this bit position
            num_full_periods = (N + 1) // period
            count += num_full_periods * (1 << k) # Each full period contributes (1<<k) ones
            
            remaining = (N + 1) % period
            count += max(0, remaining - (1 << k)) # Count ones in the partial period
            k += 1
        return count

    # Helper to calculate the number of times the 'pos'-th bit is set for numbers from 0 to N.
    # N <= 10^15. Constant time for fixed pos, or log N for max pos.
    def _count_set_bits_at_position(self, N: int, pos: int) -> int:
        if N < 0:
            return 0
        period = 1 << (pos + 1)
        num_ones_in_period = 1 << pos
        
        num_full_periods = (N + 1) // period
        count = num_full_periods * num_ones_in_period
        
        remainder = (N + 1) % period
        count += max(0, remainder - num_ones_in_period)
        return count

    # Helper to find the integer 'val' whose powerful array contains big_nums[k],
    # and the 0-indexed position 'remaining_idx' within P(val).
    # k <= 10^15. Uses binary search, log(k) calls to _count_total_elements_up_to_N.
    # Total O(log(k) * log(k)).
    def _get_val_for_index(self, k: int) -> tuple[int, int]:
        if k < 0:
            # This case indicates a conceptual "before the first element" for range sums.
            # No actual element at index -1, so val=0 means no elements from any powerful array.
            return (0, -1) 

        # Binary search for val_at_k in range [1, upper_bound]
        # Max k is 10^15. Average popcount is ~log2(N)/2. So N ~ k / (log2(k)/2)
        # 10^15 / (50/2) = 10^15 / 25. So val_at_k can be up to 10^15.
        # A safe upper bound for val_at_k is 10^15 itself, or slightly more, e.g., 2*10^15.
        low, high = 1, 10**15 + 100 # A sufficiently large upper bound for val
        val_at_k = 0
        
        while low <= high:
            mid = low + (high - low) // 2
            # count_elements is sum of popcounts up to mid.
            # This represents the total number of elements in big_nums that come from 1 to mid.
            count_elements = self._count_total_elements_up_to_N(mid)
            
            if count_elements > k: # If sum up to mid includes index k
                val_at_k = mid
                high = mid - 1
            else: # If sum up to mid does not include index k
                low = mid + 1
        
        # After binary search, val_at_k is the smallest integer such that C(val_at_k) > k.
        # This means big_nums[k] is part of the powerful array for val_at_k.
        # The index within big_nums is k.
        # Number of elements before P(val_at_k) is C(val_at_k - 1).
        elements_before_val_at_k = self._count_total_elements_up_to_N(val_at_k - 1)
        remaining_idx = k - elements_before_val_at_k
        
        return (val_at_k, remaining_idx)

    # Helper to count how many times 2^p appears in big_nums[0...k].
    # k <= 10^15, p <= 60.
    # O(log(k)^2) dominated by _get_val_for_index.
    def _count_power_p_up_to_k(self, k: int, p: int) -> int:
        if k < 0:
            return 0 # No elements if index is negative

        val_k, remaining_idx = self._get_val_for_index(k)
        
        # Count occurrences of 2^p in full powerful arrays for 1 to (val_k - 1)
        total_count = self._count_set_bits_at_position(val_k - 1, p)
        
        # Now, check the partial powerful array for val_k (elements P(val_k)[0 ... remaining_idx])
        # Does val_k contain 2^p? (i.e., is p-th bit set in val_k?)
        if (val_k >> p) & 1:
            # If it does, is this 2^p within the first (remaining_idx + 1) elements of P(val_k)?
            # The elements of P(val_k) are 2^j for set bits j in increasing order.
            # So, offset_of_bit_p_in_val_k is the number of set bits in val_k that are less than p.
            offset_of_bit_p_in_val_k = self._popcount(val_k & ((1 << p) - 1))
            
            if offset_of_bit_p_in_val_k <= remaining_idx:
                total_count += 1
                
        return total_count

    # Helper to calculate Euler's totient function phi(n).
    # n <= 10^5. O(sqrt(n)).
    def _totient_phi(self, n: int) -> int:
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1: # Remaining n is a prime factor
            result -= result // n
        return result

    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        results = []
        
        for from_i, to_i, mod_i in queries:
            if mod_i == 1:
                results.append(0) # Any number % 1 is 0
                continue

            phi_val = self._totient_phi(mod_i)
            
            # P_sum_total stores the sum of exponents.
            # Python's arbitrary precision integers handle large values for P_sum_total
            # (which can be up to ~60 * 10^15).
            P_sum_total = 0
            
            # Iterate through all possible bit positions (0 to ~60)
            # 2^60 is > 10^18, covers numbers up to 10^15.
            for p in range(61): 
                # Count occurrences of 2^p in big_nums[from_i ... to_i]
                count_p_in_range = self._count_power_p_up_to_k(to_i, p) - \
                                   self._count_power_p_up_to_k(from_i - 1, p)
                
                P_sum_total += p * count_p_in_range
            
            # Apply Euler's Totient Theorem: a^b = a^(b % phi(m) + phi(m)) (mod m) for b >= phi(m)
            # If b < phi(m), it's just a^b (mod m).
            # The `+ phi(m)` term handles cases where `b % phi(m)` results in 0, but `b` was actually >= phi(m)
            # or a multiple, ensuring the exponent is large enough for the cycle property to apply.
            final_exponent = P_sum_total % phi_val
            if P_sum_total >= phi_val: # This condition activates the +phi(m) part of the theorem.
                final_exponent += phi_val
            
            # Calculate 2^(final_exponent) % mod_i
            # pow(base, exp, mod) is efficient modular exponentiation.
            current_product = pow(2, final_exponent, mod_i)
            results.append(current_product)
            
        return results