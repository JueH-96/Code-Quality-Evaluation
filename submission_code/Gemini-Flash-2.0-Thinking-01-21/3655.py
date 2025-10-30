class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        if is_prime(n) or is_prime(m):
            return -1

        if n == m:
            return n

        q = [(n, n)]
        visited = {n}

        while q:
            curr_n, cost = q.pop(0)

            if curr_n == m:
                return cost

            s_n = str(curr_n)
            digits = [int(d) for d in s_n]
            num_digits = len(digits)

            for i in range(num_digits):
                original_digit = digits[i]

                # Increment
                if digits[i] < 9:
                    digits[i] += 1
                    next_n_inc = int("".join(map(str, digits)))
                    if not is_prime(next_n_inc) and next_n_inc not in visited:
                        visited.add(next_n_inc)
                        q.append((next_n_inc, cost + next_n_inc))
                    digits[i] = original_digit  # Backtrack

                # Decrement
                if digits[i] > 0:
                    digits[i] -= 1
                    next_n_dec = int("".join(map(str, digits)))

                    if not is_prime(next_n_dec) and next_n_dec not in visited:
                        visited.add(next_n_dec)
                        q.append((next_n_dec, cost + next_n_dec))
                    digits[i] = original_digit  # Backtrack
        return -1