from itertools import product, permutations
from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        memo = set()
        valid_multisets = []
        
        if n % 2 == 0:
            m = n // 2
            for combo in product(range(10), repeat=m):
                counts = [0] * 10
                for d in combo:
                    counts[d] += 2
                has_non_zero = any(counts[d] >= 2 for d in range(1, 10))
                if not has_non_zero:
                    continue
                key = tuple(counts)
                if key not in memo:
                    memo.add(key)
                    if self.check_divisible_even(combo, k):
                        valid_multisets.append(counts.copy())
        else:
            m_half = (n - 1) // 2
            for d_mid in range(10):
                for c_mid in range(1, n + 1, 2):
                    sum_remaining = n - c_mid
                    if sum_remaining < 0 or sum_remaining % 2 != 0:
                        continue
                    m_pairs = sum_remaining // 2
                    allowed_digits = [d for d in range(10) if d != d_mid]
                    for combo in product(allowed_digits, repeat=m_pairs):
                        counts = [0] * 10
                        counts[d_mid] = c_mid
                        for d in combo:
                            counts[d] += 2
                        if n == 1:
                            if d_mid != 0 and d_mid % k == 0:
                                key = tuple(counts)
                                if key not in memo:
                                    memo.add(key)
                                    valid_multisets.append(counts.copy())
                            continue
                        has_non_zero = any(counts[d] >= 2 for d in range(1, 10))
                        if not has_non_zero:
                            continue
                        key = tuple(counts)
                        if key not in memo:
                            memo.add(key)
                            if self.check_divisible_odd(combo, d_mid, k):
                                valid_multisets.append(counts.copy())
        
        total = 0
        for counts in valid_multisets:
            total_perm = factorial(n)
            for c in counts:
                total_perm //= factorial(c)
            leading_zero_perm = 0
            if counts[0] > 0:
                new_counts = counts.copy()
                new_counts[0] -= 1
                if new_counts[0] < 0:
                    leading_zero_perm = 0
                else:
                    leading_zero_perm = factorial(n - 1)
                    for c in new_counts:
                        leading_zero_perm //= factorial(c)
            valid = total_perm - leading_zero_perm
            total += valid
        return total
    
    def check_divisible_even(self, combo, k):
        seen = set()
        for perm in permutations(combo):
            if perm in seen:
                continue
            seen.add(perm)
            mod_val = 0
            for d in perm:
                mod_val = (mod_val * 10 + d) % k
            for d in reversed(perm):
                mod_val = (mod_val * 10 + d) % k
            if mod_val % k == 0:
                return True
        return False
    
    def check_divisible_odd(self, combo, d_mid, k):
        seen = set()
        for perm in permutations(combo):
            if perm in seen:
                continue
            seen.add(perm)
            mod_val = 0
            for d in perm:
                mod_val = (mod_val * 10 + d) % k
            mod_val = (mod_val * 10 + d_mid) % k
            for d in reversed(perm):
                mod_val = (mod_val * 10 + d) % k
            if mod_val % k == 0:
                return True
        return False