import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        total = 0
        combinations = self.generate_all_combinations(n)
        for counts in combinations:
            count_odds = sum(1 for cnt in counts if cnt % 2 == 1)
            valid = False
            if n % 2 == 0:
                if count_odds == 0:
                    valid = True
            else:
                if count_odds == 1:
                    valid = True
            if not valid:
                continue
            
            if self.is_valid_multiset(counts, n, k):
                valid_perms = self.compute_valid_permutations(counts, n)
                total += valid_perms
        return total

    def generate_all_combinations(self, n, digits=10):
        combinations = []
        def helper(idx, curr_counts, remaining):
            if idx == digits - 1:
                curr_counts.append(remaining)
                combinations.append(list(curr_counts))
                curr_counts.pop()
                return
            for i in range(remaining + 1):
                curr_counts.append(i)
                helper(idx + 1, curr_counts, remaining - i)
                curr_counts.pop()
        helper(0, [], n)
        return combinations

    def is_valid_multiset(self, counts, n, k):
        left_half_list = []
        center_d = -1
        if n % 2 == 0:
            for d in range(10):
                left_half_list.extend([d] * (counts[d] // 2))
        else:
            center_d = [d for d in range(10) if counts[d] % 2 == 1][0]
            for d in range(10):
                left_half_list.extend([d] * (counts[d] // 2))
        
        permutations = self.unique_permutations(left_half_list)
        for perm in permutations:
            if n % 2 == 0:
                p_digits = list(perm) + list(reversed(perm))
            else:
                p_digits = list(perm) + [center_d] + list(reversed(perm))
            
            if p_digits[0] == 0:
                continue
            
            num = int(''.join(map(str, p_digits)))
            if num % k == 0:
                return True
        return False

    def unique_permutations(self, elements):
        elements = sorted(elements)
        result = []
        used = [False] * len(elements)

        def backtrack(path):
            if len(path) == len(elements):
                yield list(path)
                return
            for i in range(len(elements)):
                if used[i]:
                    continue
                if i > 0 and elements[i] == elements[i - 1] and not used[i - 1]:
                    continue
                used[i] = True
                path.append(elements[i])
                yield from backtrack(path)
                path.pop()
                used[i] = False
        return list(backtrack([]))

    def compute_valid_permutations(self, counts, n):
        from math import factorial
        count_dict = {d: counts[d] for d in range(10) if counts[d] > 0}
        total = factorial(n)
        for c in count_dict.values():
            total //= factorial(c)
        
        if counts[0] == 0:
            return total
        
        new_counts = [counts[d] for d in range(10)]
        new_counts[0] -= 1
        if new_counts[0] < 0:
            invalid = 0
        else:
            denom = 1
            valid = True
            for c in new_counts:
                if c < 0:
                    valid = False
                    break
                denom *= factorial(c)
            if valid and denom != 0:
                invalid = factorial(n - 1) // denom
            else:
                invalid = 0
        return total - invalid