from collections import deque
import heapq

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return True  # Technically not prime, but for our constraint, treat as such
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        if is_prime(n):
            return -1

        if n == m:
            return n

        str_n = str(n)
        str_m = str(m)
        num_digits = len(str_n)

        costs = {}
        pq = [(n, n)]  # (current_num, current_cost)
        costs[n] = n

        while pq:
            current_num, current_cost = heapq.heappop(pq)

            if current_num == m:
                return current_cost

            s_current_num = str(current_num)
            if len(s_current_num) < num_digits:
                s_current_num = '0' * (num_digits - len(s_current_num)) + s_current_num

            for i in range(num_digits):
                original_digit = int(s_current_num[i])

                # Increase digit
                if original_digit < 9:
                    new_digit = original_digit + 1
                    new_s_num_list = list(s_current_num)
                    new_s_num_list[i] = str(new_digit)
                    next_num = int("".join(new_s_num_list))
                    if not is_prime(next_num):
                        new_cost = current_cost + next_num
                        if next_num not in costs or new_cost < costs[next_num]:
                            costs[next_num] = new_cost
                            heapq.heappush(pq, (next_num, new_cost))

                # Decrease digit
                if original_digit > 0:
                    new_digit = original_digit - 1
                    new_s_num_list = list(s_current_num)
                    new_s_num_list[i] = str(new_digit)
                    next_num = int("".join(new_s_num_list))
                    if not is_prime(next_num):
                        new_cost = current_cost + next_num
                        if next_num not in costs or new_cost < costs[next_num]:
                            costs[next_num] = new_cost
                            heapq.heappush(pq, (next_num, new_cost))

        return -1