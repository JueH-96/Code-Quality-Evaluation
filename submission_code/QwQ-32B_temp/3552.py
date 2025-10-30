class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            # Find the largest single-digit number divisible by k
            for num in range(9, 0, -1):
                if num % k == 0:
                    return str(num)
            return "0"  # This line is theoretically unreachable given constraints

        m = (n + 1) // 2
        min_prefix = '1' + '0' * (m - 1)
        current_prefix = '9' * m

        while True:
            mod = self.compute_mod(current_prefix, n, k)
            if mod == 0:
                return self.build_palindrome(current_prefix, n)
            new_prefix = self.decrement(current_prefix)
            if new_prefix is None:
                break
            current_prefix = new_prefix

        # Check the minimum prefix in case the loop missed it
        mod_min = self.compute_mod(min_prefix, n, k)
        if mod_min == 0:
            return self.build_palindrome(min_prefix, n)
        else:
            # According to problem constraints, there must be a solution
            return self.build_palindrome(min_prefix, n)

    def compute_mod(self, prefix: str, n: int, k: int) -> int:
        m = len(prefix)
        is_even = (n % 2 == 0)
        if is_even:
            p_mod = 0
            for c in prefix:
                p_mod = (p_mod * 10 + int(c)) % k
            rev_mod = 0
            for c in reversed(prefix):
                rev_mod = (rev_mod * 10 + int(c)) % k
            pow_10_m = pow(10, m, k)
            total_mod = (p_mod * pow_10_m + rev_mod) % k
        else:
            p_mod = 0
            for c in prefix:
                p_mod = (p_mod * 10 + int(c)) % k
            rev_part_mod = 0
            for c in reversed(prefix[:-1]):
                rev_part_mod = (rev_part_mod * 10 + int(c)) % k
            pow_10_m_minus_1 = pow(10, m - 1, k)
            total_mod = (p_mod * pow_10_m_minus_1 + rev_part_mod) % k
        return total_mod

    def decrement(self, s: str) -> str:
        s_list = list(s)
        i = len(s_list) - 1
        while i >= 0:
            if s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            else:
                s_list[i] = str(int(s_list[i]) - 1)
                break
        if s_list[0] == '0':
            return None
        return ''.join(s_list)

    def build_palindrome(self, prefix: str, n: int) -> str:
        if n % 2 == 0:
            return prefix + prefix[::-1]
        else:
            return prefix + prefix[:-1][::-1]