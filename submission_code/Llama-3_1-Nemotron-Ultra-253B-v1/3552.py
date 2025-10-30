class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        L = (n + 1) // 2
        prefix = '9' * L

        def compute_remainder(prefix: str) -> int:
            if n % 2 == 0:
                p_mod = 0
                for c in prefix:
                    p_mod = (p_mod * 10 + int(c)) % k
                rev_p_mod = 0
                for c in reversed(prefix):
                    rev_p_mod = (rev_p_mod * 10 + int(c)) % k
                a = pow(10, L, k)
                total_mod = (p_mod * a + rev_p_mod) % k
            else:
                p_mod = 0
                for c in prefix:
                    p_mod = (p_mod * 10 + int(c)) % k
                rev_part = prefix[:-1]
                rev_p_mod = 0
                for c in reversed(rev_part):
                    rev_p_mod = (rev_p_mod * 10 + int(c)) % k
                a = pow(10, len(rev_part), k)
                total_mod = (p_mod * a + rev_p_mod) % k
            return total_mod

        def decrement_prefix(prefix: str) -> str:
            prefix_list = list(prefix)
            i = len(prefix_list) - 1
            while i >= 0 and prefix_list[i] == '0':
                prefix_list[i] = '9'
                i -= 1
            if i < 0:
                return None
            prefix_list[i] = str(int(prefix_list[i]) - 1)
            if prefix_list[0] == '0':
                return None
            return ''.join(prefix_list)

        while True:
            rem = compute_remainder(prefix)
            if rem == 0:
                if n % 2 == 0:
                    return prefix + prefix[::-1]
                else:
                    return prefix + prefix[:-1][::-1]
            next_prefix = decrement_prefix(prefix)
            if next_prefix is None:
                return '0'
            prefix = next_prefix