class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def subtract_int_from_str(s, num):
            s_list = list(s)
            idx = len(s_list) - 1
            while num > 0 and idx >= 0:
                digit = int(s_list[idx])
                if digit >= num:
                    s_list[idx] = str(digit - num)
                    num = 0
                else:
                    s_list[idx] = str(digit + 10 - num)
                    num = 1  # borrow from the next digit
                idx -= 1
            # Handle leading zeros
            result = ''.join(s_list).lstrip('0')
            return result if result else '0'

        n = int(n)
        k = int(k)
        len_L = (n + 1) // 2
        len_L_rev = n // 2
        pow_10_mod_k = pow(10, len_L_rev, k)
        max_L = '9' * len_L

        for i in range(k):
            L_candidate = subtract_int_from_str(max_L, i)
            if L_candidate == '0' or L_candidate[0] == '0':
                continue
            L_mod_k = 0
            for c in L_candidate:
                L_mod_k = (L_mod_k * 10 + int(c)) % k
            L_rev_mod_k = 0
            for c in reversed(L_candidate[:len_L_rev]):
                L_rev_mod_k = (L_rev_mod_k * 10 + int(c)) % k
            N_mod_k = (L_mod_k * pow_10_mod_k + L_rev_mod_k) % k
            if N_mod_k == 0:
                if n % 2 == 0:
                    palindrome = L_candidate + L_candidate[::-1]
                else:
                    palindrome = L_candidate + L_candidate[:-1][::-1]
                return palindrome
        return ''  # if no such palindrome exists