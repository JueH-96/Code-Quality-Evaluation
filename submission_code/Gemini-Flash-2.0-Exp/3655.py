class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        def solve():
            q = [(n, 0, [n])]
            visited = {n}
            
            while q:
                curr_num, cost, path = q.pop(0)
                
                if curr_num == m:
                    return sum(path)

                s_curr_num = str(curr_num)
                for i in range(len(s_curr_num)):
                    digit = int(s_curr_num[i])

                    # Increase digit
                    if digit < 9:
                        new_digit = digit + 1
                        new_num_str = list(s_curr_num)
                        new_num_str[i] = str(new_digit)
                        new_num = int("".join(new_num_str))
                        
                        if not is_prime(new_num) and new_num not in visited:
                            visited.add(new_num)
                            q.append((new_num, cost + new_num, path + [new_num]))

                    # Decrease digit
                    if digit > 0:
                        new_digit = digit - 1
                        new_num_str = list(s_curr_num)
                        new_num_str[i] = str(new_digit)
                        new_num = int("".join(new_num_str))

                        if not is_prime(new_num) and new_num not in visited:
                            visited.add(new_num)
                            q.append((new_num, cost + new_num, path + [new_num]))
            return -1

        if is_prime(n) or is_prime(m):
            return -1

        return solve()