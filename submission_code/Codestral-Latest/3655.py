class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def is_prime(num):
            if num <= 1:
                return False
            if num <= 3:
                return True
            if num % 2 == 0 or num % 3 == 0:
                return False
            i = 5
            while i * i <= num:
                if num % i == 0 or num % (i + 2) == 0:
                    return False
                i += 6
            return True

        def bfs(start, target):
            from collections import deque

            queue = deque([(start, 0)])
            visited = set()
            visited.add(start)

            while queue:
                current, cost = queue.popleft()
                if current == target:
                    return cost

                current_str = str(current)
                for i in range(len(current_str)):
                    if current_str[i] != '9':
                        new_num = int(current_str[:i] + str(int(current_str[i]) + 1) + current_str[i+1:])
                        if not is_prime(new_num) and new_num not in visited:
                            visited.add(new_num)
                            queue.append((new_num, cost + new_num))
                    if current_str[i] != '0':
                        new_num = int(current_str[:i] + str(int(current_str[i]) - 1) + current_str[i+1:])
                        if not is_prime(new_num) and new_num not in visited:
                            visited.add(new_num)
                            queue.append((new_num, cost + new_num))

            return -1

        if is_prime(m):
            return -1

        return bfs(n, m)