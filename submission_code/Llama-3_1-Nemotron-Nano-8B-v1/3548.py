class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from math import factorial
        from itertools import permutations

        def generate_even_distributions(m):
            distributions = []
            def backtrack(index, current, remaining):
                if index == 10:
                    if remaining == 0:
                        distributions.append(current.copy())
                    return
                for i in range(0, remaining + 1):
                    current.append(i)
                    backtrack(index + 1, current, remaining - i)
                    current.pop()
            backtrack(0, [], m)
            return distributions

        def generate_odd_distributions(n):
            m = (n - 1) // 2
            distributions = []
            for d in range(10):
                for y in range(0, m + 1):
                    remaining = m - y
                    other_distributions = []
                    def backtrack_other(index, current, rem):
                        if index == 9:
                            if rem == 0:
                                other_distributions.append(current.copy())
                            return
                        for i in range(0, rem + 1):
                            current.append(i)
                            backtrack_other(index + 1, current, rem - i)
                            current.pop()
                    backtrack_other(0, [], remaining)
                    for od in other_distributions:
                        counts = [0] * 10
                        counts[d] = 2 * y + 1
                        other_digits = [i for i in range(10) if i != d]
                        for idx in range(9):
                            digit = other_digits[idx]
                            counts[digit] = 2 * od[idx]
                        distributions.append(counts)
            return distributions

        def is_valid_distribution(counts, n, k):
            if n % 2 == 0:
                first_half = []
                for d in range(10):
                    cnt = counts[d] // 2
                    first_half.extend([d] * cnt)
                perms = set(permutations(first_half))
                for perm in perms:
                    if not perm:
                        continue
                    if perm[0] == 0:
                        continue
                    palindrome = list(perm) + list(reversed(perm))
                    num = int(''.join(map(str, palindrome)))
                    if num % k == 0:
                        return True
                return False
            else:
                middle_digit = None
                for d in range(10):
                    if counts[d] % 2 == 1:
                        middle_digit = d
                        break
                first_half = []
                for d in range(10):
                    if d == middle_digit:
                        cnt = (counts[d] - 1) // 2
                    else:
                        cnt = counts[d] // 2
                    first_half.extend([d] * cnt)
                perms = set(permutations(first_half))
                for perm in perms:
                    if not perm:
                        num = middle_digit
                        if num == 0:
                            continue
                        if num % k == 0:
                            return True
                    else:
                        if perm[0] == 0:
                            continue
                        palindrome = list(perm) + [middle_digit] + list(reversed(perm))
                        num = int(''.join(map(str, palindrome)))
                        if num % k == 0:
                            return True
                return False

        total = 0
        if n % 2 == 0:
            m = n // 2
            even_distributions = generate_even_distributions(m)
            distributions = even_distributions
        else:
            distributions = generate_odd_distributions(n)

        for counts in distributions:
            if not is_valid_distribution(counts, n, k):
                continue
            perm_total = factorial(n)
            for d in range(10):
                perm_total //= factorial(counts[d])
            if counts[0] > 0:
                invalid = factorial(n - 1)
                invalid //= factorial(counts[0] - 1)
                for d in range(1, 10):
                    invalid //= factorial(counts[d])
                valid = perm_total - invalid
            else:
                valid = perm_total
            total += valid
        return total