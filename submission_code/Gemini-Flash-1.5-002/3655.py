class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        q = [(n, 0, [n])]  # (current_num, cost, path)
        visited = {n}

        while q:
            curr_num, cost, path = q.pop(0)
            if curr_num == m:
                return cost + curr_num

            s_curr = str(curr_num)
            for i in range(len(s_curr)):
                digit = int(s_curr[i])
                if digit < 9:
                    next_num = int(s_curr[:i] + str(digit + 1) + s_curr[i+1:])
                    if not is_prime(next_num) and next_num not in visited:
                        q.append((next_num, cost + curr_num, path + [next_num]))
                        visited.add(next_num)
                if digit > 0:
                    next_num = int(s_curr[:i] + str(digit - 1) + s_curr[i+1:])
                    if not is_prime(next_num) and next_num not in visited:
                        q.append((next_num, cost + curr_num, path + [next_num]))
                        visited.add(next_num)

        return -1