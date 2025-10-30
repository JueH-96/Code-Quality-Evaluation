import math

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num <= 1:
                return True
            if num <= 3:
                return False
            if num % 2 == 0 or num % 3 == 0:
                return True
            i = 5
            while i * i <= num:
                if num % i == 0 or num % (i + 2) == 0:
                    return True
                i += 6
            return False

        if is_prime(m):
            return -1

        q = [(n, n)]
        cost_track = {n: n}

        while q:
            curr_num, curr_cost = q.pop(0)

            if curr_num == m:
                return curr_cost

            if is_prime(curr_num):
                continue

            s_curr_num = str(curr_num)
            num_digits = len(s_curr_num)

            for i in range(num_digits):
                digit = int(s_curr_num[i])

                # Increment
                if digit < 9:
                    next_digit_inc = digit + 1
                    next_num_str_inc = list(s_curr_num)
                    next_num_str_inc[i] = str(next_digit_inc)
                    next_num_inc = int("".join(next_num_str_inc))

                    if not is_prime(next_num_inc):
                        if next_num_inc not in cost_track or curr_cost + next_num_inc < cost_track[next_num_inc]:
                            cost_track[next_num_inc] = curr_cost + next_num_inc
                            q.append((next_num_inc, cost_track[next_num_inc]))

                # Decrement
                if digit > 0:
                    next_digit_dec = digit - 1
                    next_num_str_dec = list(s_curr_num)
                    next_num_str_dec[i] = str(next_digit_dec)
                    next_num_dec = int("".join(next_num_str_dec))

                    if not is_prime(next_num_dec):
                        if next_num_dec not in cost_track or curr_cost + next_num_dec < cost_track[next_num_dec]:
                            cost_track[next_num_dec] = curr_cost + next_num_dec
                            q.append((next_num_dec, cost_track[next_num_dec]))
        return -1