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

        if is_prime(n) or is_prime(m):
            return -1

        q = [(n, 0)]
        visited = {n}
        while q:
            curr_n, curr_cost = q.pop(0)
            if curr_n == m:
                return curr_cost

            s_n = str(curr_n)
            for i in range(len(s_n)):
                digit = int(s_n[i])
                if digit < 9:
                    new_s_n = list(s_n)
                    new_s_n[i] = str(digit + 1)
                    next_n = int("".join(new_s_n))
                    if not is_prime(next_n) and next_n not in visited:
                        visited.add(next_n)
                        q.append((next_n, curr_cost + next_n))

                if digit > 0:
                    new_s_n = list(s_n)
                    new_s_n[i] = str(digit - 1)
                    next_n = int("".join(new_s_n))

                    if not is_prime(next_n) and next_n not in visited:
                        visited.add(next_n)
                        q.append((next_n, curr_cost + next_n))
        
        return -1