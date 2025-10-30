class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(num_str):
            return num_str == num_str[::-1]

        def is_k_palindromic(num, k):
            return num % k == 0 and is_palindrome(str(num))

        def can_rearrange_to_k_palindromic(num_str, k):
            counts = {}
            for digit in num_str:
                counts[digit] = counts.get(digit, 0) + 1

            for i in range(10**(n-1), 10**n):
                if is_k_palindromic(i, k):
                    i_str = str(i)
                    i_counts = {}
                    for digit in i_str:
                        i_counts[digit] = i_counts.get(digit, 0) + 1

                    
                    if len(counts) != len(i_counts):
                        continue

                    possible = True
                    for digit, count in counts.items():
                        if digit not in i_counts or i_counts[digit] != count:
                            possible = False
                            break
                    
                    if possible:
                        return True
            return False

        count = 0
        for i in range(10**(n-1), 10**n):
            if can_rearrange_to_k_palindromic(str(i), k):
                count += 1
        return count