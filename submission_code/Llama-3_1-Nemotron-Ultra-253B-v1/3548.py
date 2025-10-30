from itertools import permutations
from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def generate_tuples(m):
            result = []
            def helper(remaining, length, current):
                if length == 0:
                    if remaining == 0:
                        result.append(current.copy())
                    return
                for i in range(remaining + 1):
                    current.append(i)
                    helper(remaining - i, length - 1, current)
                    current.pop()
            helper(m, 10, [])
            return result

        total = 0

        if n % 2 == 0:
            m = n // 2
            for c_tuple in generate_tuples(m):
                if sum(c_tuple[1:]) == 0:
                    continue
                counts = [2 * count for count in c_tuple]
                digits = []
                for i in range(10):
                    digits.extend([i] * (counts[i] // 2))
                seen = set()
                has_valid = False
                for p in permutations(digits):
                    if p[0] == 0:
                        continue
                    if p not in seen:
                        seen.add(p)
                        num_str = ''.join(map(str, p + p[::-1]))
                        num = int(num_str)
                        if num % k == 0:
                            has_valid = True
                            break
                    if has_valid:
                        break
                if has_valid:
                    total += self.count_original(counts)
        else:
            m = (n - 1) // 2
            for d in range(10):
                for c_tuple in generate_tuples(m):
                    c = c_tuple
                    if d == 0 and sum(c[1:]) < 1:
                        continue
                    counts = [2 * count for count in c]
                    counts[d] += 1
                    sum_non_zero = sum(counts[1:])
                    if sum_non_zero < 1:
                        continue
                    middle_digit = d
                    digits = []
                    for i in range(10):
                        if i == middle_digit:
                            continue
                        digits.extend([i] * (counts[i] // 2))
                    seen = set()
                    has_valid = False
                    for p in permutations(digits):
                        if p[0] == 0:
                            continue
                        if p not in seen:
                            seen.add(p)
                            num_str = ''.join(map(str, p + [middle_digit] + p[::-1]))
                            num = int(num_str)
                            if num % k == 0:
                                has_valid = True
                                break
                        if has_valid:
                            break
                    if has_valid:
                        total += self.count_original(counts)
        return total

    def count_original(self, counts):
        n = sum(counts)
        total = factorial(n)
        for c in counts:
            total //= factorial(c)
        if counts[0] == 0:
            return total
        new_counts = counts.copy()
        new_counts[0] -= 1
        perms_with_zero = factorial(n - 1)
        for c in new_counts:
            perms_with_zero //= factorial(c)
        return total - perms_with_zero