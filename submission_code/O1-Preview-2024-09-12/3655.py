class Solution:
    def minOperations(self, n: int, m: int) -> int:
        import sys
        import heapq

        # First, check if initial n or final m is prime
        max_num = max(n, m, 9999)

        # Sieve of Eratosthenes to precompute is_prime up to max_num
        is_prime = [False, False] + [True] * (max_num -1)
        for i in range(2, int(max_num ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, max_num +1, i):
                    is_prime[j] = False

        if is_prime[n] or is_prime[m]:
            return -1  # Cannot start from or reach a prime number

        n_str = str(n)
        m_str = str(m)
        if len(n_str) != len(m_str):
            return -1  # Numbers must have the same number of digits

        from collections import defaultdict

        total_cost = defaultdict(lambda: sys.maxsize)
        total_cost[n] = n

        heap = [(n, n)]  # (total_cost, current_n)
        while heap:
            cost, current_n = heapq.heappop(heap)

            if current_n == m:
                return cost

            if cost > total_cost[current_n]:
                continue  # We have already found a better path to current_n

            current_n_str = str(current_n)
            len_n = len(current_n_str)

            # Generate neighbors
            for i in range(len_n):
                digit = current_n_str[i]
                digit_int = int(digit)

                # Try to increase the digit (if not '9')
                if digit != '9':
                    new_digit = str(digit_int + 1)
                    next_n_str = (
                        current_n_str[:i] + new_digit + current_n_str[i+1:]
                    )
                    # Ensure first digit is not '0'
                    if next_n_str[0] != '0':
                        next_n = int(next_n_str)
                        if not is_prime[next_n]:
                            new_cost = cost + next_n
                            if new_cost < total_cost[next_n]:
                                total_cost[next_n] = new_cost
                                heapq.heappush(heap, (new_cost, next_n))
                # Try to decrease the digit (if not '0')
                if digit != '0':
                    # Cannot decrease first digit to '0'
                    if i != 0 or digit != '1':
                        new_digit = str(digit_int -1)
                        next_n_str = (
                            current_n_str[:i] + new_digit + current_n_str[i+1:]
                        )
                        # Ensure first digit is not '0'
                        if next_n_str[0] != '0':
                            next_n = int(next_n_str)
                            if not is_prime[next_n]:
                                new_cost = cost + next_n
                                if new_cost < total_cost[next_n]:
                                    total_cost[next_n] = new_cost
                                    heapq.heappush(heap, (new_cost, next_n))
        # If we reach here, m is unreachable
        return -1