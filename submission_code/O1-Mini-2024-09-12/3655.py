class Solution:
    def minOperations(self, n: int, m: int) -> int:
        from collections import deque
        import math

        def is_prime(num):
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(math.sqrt(num)) + 1, 2):
                if num % i == 0:
                    return False
            return True

        # Check if n or m have different number of digits
        if len(str(n)) != len(str(m)):
            return -1

        # Check if n or m is prime
        if is_prime(n) or is_prime(m):
            return -1

        from heapq import heappush, heappop

        heap = []
        heappush(heap, (n, n))  # (cumulative_cost, current_number)
        visited = {}

        while heap:
            cost, current = heappop(heap)

            if current == m:
                return cost

            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost

            str_num = str(current)
            for i in range(len(str_num)):
                digit = int(str_num[i])

                # Increase digit
                if digit < 9:
                    new_digit = digit + 1
                    new_num = int(str_num[:i] + str(new_digit) + str_num[i+1:])
                    if not is_prime(new_num):
                        heappush(heap, (cost + new_num, new_num))

                # Decrease digit
                if digit > 0:
                    new_digit = digit - 1
                    new_num = int(str_num[:i] + str(new_digit) + str_num[i+1:])
                    if not is_prime(new_num):
                        heappush(heap, (cost + new_num, new_num))

        return -1