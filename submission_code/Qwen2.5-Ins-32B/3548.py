class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from itertools import permutations
        
        def is_k_palindromic(x: int, k: int) -> bool:
            return str(x) == str(x)[::-1] and x % k == 0
        
        def generate_numbers(length: int) -> set:
            if length == 1:
                return set(range(1, 10))
            else:
                prev_numbers = generate_numbers(length - 1)
                new_numbers = set()
                for num in prev_numbers:
                    for digit in range(10):
                        new_numbers.add(num * 10 + digit)
                return new_numbers
        
        def count_good_integers(n: int, k: int) -> int:
            count = 0
            numbers = generate_numbers(n)
            for num in numbers:
                str_num = str(num)
                for perm in set(permutations(str_num)):
                    perm_num = int(''.join(perm))
                    if is_k_palindromic(perm_num, k):
                        count += 1
                        break
            return count
        
        return count_good_integers(n, k)