class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        """
        Finds the largest n-digit k-palindromic number.

        A k-palindromic number is a palindrome that is divisible by k.
        The approach is to construct the number greedily from left to right.
        A palindrome is determined by its first half of digits. Let the length
        of the first half be m = ceil(n/2). We want to find the lexicographically
        largest string of m digits, c_1, c_2, ..., c_m, that can form a valid
        k-palindromic number.

        The divisibility condition P % k == 0 can be written as a linear congruence
        equation involving the digits c_1, ..., c_m.
        (c_1*w_1 + c_2*w_2 + ... + c_m*w_m) (mod k) == 0
        where w_i are weights derived from powers of 10.

        To solve this greedily for c_1, c_2, ..., c_m, we need to know at each
        step if the remaining digits can form a sum that satisfies the congruence.
        This is a classic dynamic programming problem.

        1. Precompute weights w_i:
           - For a pair of digits c_i and c_{n+1-i}, the contribution is c_i * (10^{n-i} + 10^{i-1}).
           - For a middle digit c_m (if n is odd), the contribution is c_m * 10^{n-m}.
           - These are computed modulo k.

        2. DP for reachability:
           - Let possible[i][rem] be a boolean indicating if a remainder `rem`
             is achievable using the suffix of the first half from digit c_i to c_m.
           - We compute this table backwards from i = m to 1.

        3. Greedy construction:
           - Iterate from i = 1 to m to determine digits c_1, ..., c_m.
           - For each c_i, try digits d from 9 down to 0 (or 1 for c_1).
           - Choose the largest d such that the remaining congruence can be satisfied,
             which is checked using the `possible` table.
        """
        
        # m is the length of the first half of the palindrome
        m = (n + 1) // 2
        l = n // 2

        # Precompute powers of 10 modulo k
        p10 = [1] * (n + 1)
        for i in range(1, n + 1):
            p10[i] = (p10[i - 1] * 10) % k

        # w[i] is the weight of the i-th digit (1-indexed) of the first half.
        w = [0] * (m + 1)
        # For pairs of digits c_i and c_{n+1-i}
        for i in range(1, l + 1):
            w[i] = (p10[n - i] + p10[i - 1]) % k
        # For the middle digit if n is odd
        if n % 2 == 1:
            w[m] = p10[l] % k
            
        # DP to find achievable remainders
        # possible[i][rem] is true if a remainder `rem` is achievable
        # using the suffix of the first half starting at digit `i`.
        possible = [[False] * k for _ in range(m + 2)]
        possible[m + 1][0] = True

        for i in range(m, 0, -1):
            for d in range(10):
                rem_d = (d * w[i]) % k
                for rem_prev in range(k):
                    if possible[i + 1][rem_prev]:
                        new_rem = (rem_d + rem_prev) % k
                        possible[i][new_rem] = True

        # Greedily construct the first half of the palindrome
        first_half_digits = []
        current_rem = 0

        for i in range(1, m + 1):
            start_digit = 9
            end_digit = 0
            # The first digit of a number cannot be 0.
            if i == 1:
                 end_digit = 1
            
            for d in range(start_digit, end_digit - 1, -1):
                rem_d = (d * w[i]) % k
                
                # Required remainder from the rest of the digits:
                target_rem = (k - (current_rem + rem_d) % k) % k

                if possible[i + 1][target_rem]:
                    first_half_digits.append(str(d))
                    current_rem = (current_rem + rem_d) % k
                    break
            else:
                # This block should not be reached if a solution always exists.
                return ""
        
        first_half_str = "".join(first_half_digits)
        
        # Construct the full palindrome from the first half.
        # The second half is the reverse of the first `l` characters of the first half.
        second_half_str = first_half_str[:l][::-1]
        
        return first_half_str + second_half_str