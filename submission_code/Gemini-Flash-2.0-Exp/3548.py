class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        from math import factorial

        def is_k_palindromic(counts, num_digits):
            if num_digits % 2 != 0:
                odd_count = 0
                for count in counts.values():
                    if count % 2 != 0:
                        odd_count += 1
                if odd_count != 1:
                    return False
            else:
                for count in counts.values():
                    if count % 2 != 0:
                        return False

            digits = []
            for digit, count in counts.items():
                digits.extend([digit] * count)

            import itertools
            for perm in itertools.permutations(digits):
                if perm[0] == 0 and num_digits > 1:
                    continue
                num = int("".join(map(str, perm)))
                if num % k == 0:
                    s = "".join(map(str, perm))
                    if s == s[::-1]:
                        return True
            return False

        def is_good(counts, num_digits):
            if num_digits % 2 != 0:
                odd_count = 0
                for count in counts.values():
                    if count % 2 != 0:
                        odd_count += 1
                if odd_count != 1:
                    return False
            else:
                for count in counts.values():
                    if count % 2 != 0:
                        return False
            return True

        def count_permutations(counts, num_digits):
            total_count = factorial(num_digits)
            for count in counts.values():
                total_count //= factorial(count)
            return total_count

        def solve():
            count = 0
            for i in range(10**(n-1), 10**n):
                s = str(i)
                counts = Counter(map(int, list(s)))
                
                if is_good(counts, n):
                    
                    digits = []
                    for digit, cnt in counts.items():
                        digits.extend([digit] * cnt)
                    
                    import itertools
                    
                    valid_permutations = set()
                    
                    for perm in itertools.permutations(digits):
                        if perm[0] == 0 and n > 1:
                            continue
                        num = int("".join(map(str, perm)))
                        if num % k == 0:
                            s = "".join(map(str, perm))
                            if s == s[::-1]:
                                valid_permutations.add(num)
                    
                    if len(valid_permutations) > 0:
                        count += 1

            return count

        def solve2():
            count = 0
            
            def backtrack(index, current_num, counts):
                nonlocal count
                
                if index == n:
                    if current_num % k == 0 and str(current_num) == str(current_num)[::-1]:
                        count += 1
                    return
                
                for digit in range(10):
                    if counts[digit] > 0:
                        if index == 0 and digit == 0:
                            continue
                        
                        new_counts = counts.copy()
                        new_counts[digit] -= 1
                        
                        backtrack(index + 1, current_num * 10 + digit, new_counts)

            
            
            return count

        def solve3():
            
            def is_k_palindromic_num(num):
                s = str(num)
                if s != s[::-1]:
                    return False
                if num % k != 0:
                    return False
                return True

            def is_good_num(num):
                s = str(num)
                counts = Counter(s)
                
                import itertools
                
                for perm in itertools.permutations(s):
                    if perm[0] == '0' and n > 1:
                        continue
                    
                    num_perm = int("".join(perm))
                    if is_k_palindromic_num(num_perm):
                        return True
                return False
            
            count = 0
            for i in range(10**(n-1), 10**n):
                if is_good_num(i):
                    count += 1
            return count

        return solve3()