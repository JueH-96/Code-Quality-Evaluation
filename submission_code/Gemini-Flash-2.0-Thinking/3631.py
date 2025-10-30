class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        n_str = s
        n = len(n_str)
        mod = 10**9 + 7

        memo_reducible = {}
        def is_k_reducible(num, k_rem):
            if (num, k_rem) in memo_reducible:
                return memo_reducible[(num, k_rem)]
            if num == 1:
                return True
            if k_rem == 0:
                return False

            binary_repr = bin(num)[2:]
            set_bits = binary_repr.count('1')
            result = is_k_reducible(set_bits, k_rem - 1)
            memo_reducible[(num, k_rem)] = result
            return result

        memo_count = {}
        def count_numbers_with_set_bits(index, count, tight, num_bits):
            if (index, count, tight) in memo_count:
                return memo_count[(index, count, tight)]

            if count > num_bits:
                return 0

            if index == n:
                return 1 if count == num_bits else 0

            upper_bound = int(n_str[index]) if tight else 1
            res = 0
            for digit in range(upper_bound + 1):
                new_tight = tight and (digit == upper_bound)
                new_count = count + digit
                res = (res + count_numbers_with_set_bits(index + 1, new_count, new_tight, num_bits)) % mod

            memo_count[(index, count, tight)] = res
            return res

        ans = 0
        for num_set_bits in range(1, n + 1):
            memo_count.clear()
            if is_k_reducible(num_set_bits, k - 1):
                ans = (ans + count_numbers_with_set_bits(0, 0, True, num_set_bits)) % mod

        # Subtract 1 if n itself is k-reducible and was counted
        n_int = int(n_str, 2)
        binary_n = bin(n_int)[2:]

        def check_n_reducible(start_num, k_rem):
            if start_num == 1:
                return True
            if k_rem == 0:
                return False
            binary_repr = bin(start_num)[2:]
            set_bits = binary_repr.count('1')
            return check_n_reducible(set_bits, k_rem - 1)

        if n_int > 0 and check_n_reducible(n_int, k):

            memo_count.clear()
            n_set_bits = binary_n.count('1')
            if is_k_reducible(n_set_bits, k - 1):
                # We counted numbers less than or equal to n. If n is k-reducible,
                # and its reduction path leads to a set bit count that is also k-1 reducible,
                # then our count includes n. We need to subtract 1 if n > 0.
                pass # The logic already counts numbers strictly less than n

        # Handle the case where the numbers themselves are 1 and k >= 1
        if k >= 1 and n_int > 1:
            memo_count.clear()
            if is_k_reducible(1, k - 1): # Check if 1 is k-1 reducible (always true)
                # Count how many numbers less than n have 1 set bit (powers of 2)
                for i in range(n):
                    val = 1 << i
                    if val < n_int:
                        ans = (ans + 1) % mod
                    else:
                        break

        final_ans = 0
        for i in range(1, n_int):
            num = i
            steps = 0
            current = num
            reducible = False
            for _ in range(k + 1):
                if current == 1:
                    reducible = True
                    break
                binary_repr = bin(current)[2:]
                current = binary_repr.count('1')
                steps += 1
            if reducible:
                final_ans = (final_ans + 1) % mod

        return final_ans