class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num <= 1:
                return True
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        q = [(n, 0)]
        visited = {n}
        cost = 0

        while q:
            curr_n, curr_cost = q.pop(0)

            if curr_n == m:
                return curr_cost

            s_n = str(curr_n)
            s_m = str(m)

            if len(s_n) != len(s_m):
                return -1

            for i in range(len(s_n)):
                digit_n = int(s_n[i])
                digit_m = int(s_m[i])

                if digit_n < 9:
                    next_n = int(s_n[:i] + str(digit_n + 1) + s_n[i+1:])
                    if next_n not in visited and not is_prime(next_n):
                        visited.add(next_n)
                        q.append((next_n, curr_cost + next_n))
                
                if digit_n > 0:
                    next_n = int(s_n[:i] + str(digit_n - 1) + s_n[i+1:])
                    if next_n not in visited and not is_prime(next_n):
                        visited.add(next_n)
                        q.append((next_n, curr_cost + next_n))

        return -1